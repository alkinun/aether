//! Runtime environment resolution + a streamed, animatable `cargo build` of the
//! training client, and the final `exec` into that client.

use crate::config;
use anyhow::Result;
use std::{
    io::{BufRead, BufReader},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{Arc, Mutex},
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

// --- Environment -----------------------------------------------------------

/// Resolved sandbox/system environment used both to build the client and to run
/// it. Prefers the installer's sandbox under `.aethercompute/` and falls back to
/// the user's system toolchain when the sandbox isn't present.
pub struct Env {
    pub cargo: PathBuf,
    pub rustup_home: Option<PathBuf>,
    pub cargo_home: Option<PathBuf>,
    /// All directories that must be on the loader path for libtorch to resolve:
    /// `<torch>/lib` plus every `nvidia/*/lib` from the CUDA pip wheels.
    pub torch_lib_dirs: Vec<PathBuf>,
}

impl Env {
    pub fn detect() -> Self {
        let cargo_home = config::sandbox_cargo_home();
        let rustup_home = config::sandbox_rustup_home();

        let sandboxed_cargo = cargo_home.join("bin").join("cargo");
        let cargo = if sandboxed_cargo.exists() {
            sandboxed_cargo
        } else {
            PathBuf::from("cargo")
        };

        let rustup_home = rustup_home.exists().then_some(rustup_home);
        let cargo_home = cargo_home.exists().then_some(cargo_home);

        let torch_lib_dirs = detect_torch_lib_dirs();

        Self {
            cargo,
            rustup_home,
            cargo_home,
            torch_lib_dirs,
        }
    }

    /// Apply the full sandbox + libtorch environment to a command. Used for
    /// cargo builds (torch-sys needs libtorch at build time) and for running
    /// the client (it needs libtorch at run time).
    pub fn apply(&self, cmd: &mut Command) {
        if let Some(h) = &self.rustup_home {
            cmd.env("RUSTUP_HOME", h);
        }
        if let Some(h) = &self.cargo_home {
            cmd.env("CARGO_HOME", h);
        }
        cmd.env("LIBTORCH_USE_PYTORCH", "1")
            .env("LIBTORCH_BYPASS_VERSION_CHECK", "1")
            .env("RUST_MIN_STACK", "268435456");
        // Prepend every torch-related lib dir so libtorch_cuda and the CUDA
        // runtime/cublas/etc. from the nvidia pip packages all resolve.
        for lib in &self.torch_lib_dirs {
            prepend_library_path(cmd, "LD_LIBRARY_PATH", lib);
            prepend_library_path(cmd, "DYLD_LIBRARY_PATH", lib);
        }
    }
}

fn prepend_library_path(cmd: &mut Command, var: &str, entry: &std::path::Path) {
    let new = match std::env::var(var) {
        Ok(existing) if !existing.is_empty() => {
            format!("{}:{}", entry.to_string_lossy(), existing)
        }
        _ => entry.to_string_lossy().into_owned(),
    };
    cmd.env(var, new);
}

/// Locate every lib dir libtorch needs. Collects `<torch>/lib` plus each
/// `nvidia/*/lib` shipped by the CUDA pip wheels (libcudart, libcublas, …).
///
/// Prefer the sandbox venv over system Python. torch-sys links against libtorch
/// C++ symbols, and arbitrary system torch versions can differ at link time.
fn detect_torch_lib_dirs() -> Vec<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    let venv_py = config::sandbox_venv().join("bin").join("python");
    if venv_py.exists() && !candidates.contains(&venv_py) {
        candidates.push(venv_py);
    }
    if let Some(p) = which("python3").or_else(|| which("python")) {
        candidates.push(p);
    }
    for python in &candidates {
        if let Some(dirs) = probe_torch_lib_dirs(python) {
            return dirs;
        }
    }
    Vec::new()
}

/// Run the dir-collecting snippet against one python; returns `Some` only when
/// that python can actually `import torch`.
fn probe_torch_lib_dirs(python: &PathBuf) -> Option<Vec<PathBuf>> {
    let script = "import pathlib\n\
try:\n    import torch\nexcept Exception:\n    raise SystemExit(1)\n\
torch_file = pathlib.Path(torch.__file__).resolve()\n\
dirs = [str(torch_file.parent / 'lib')]\n\
site = torch_file.parent.parent\n\
nv = site / 'nvidia'\n\
if nv.is_dir():\n    dirs += [str(d) for d in sorted(nv.glob('*/lib'))]\n\
print(':'.join(dirs))";
    let out = Command::new(python).arg("-c").arg(script).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let dirs: Vec<PathBuf> = s
        .trim()
        .split(':')
        .filter(|d| !d.is_empty())
        .map(PathBuf::from)
        .filter(|p| p.is_dir())
        .collect();
    if dirs.is_empty() {
        None
    } else {
        Some(dirs)
    }
}

fn which(bin: &str) -> Option<PathBuf> {
    let out = Command::new("sh")
        .arg("-c")
        .arg(format!("command -v {bin}"))
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let p = PathBuf::from(s.trim());
    p.exists().then_some(p)
}

// --- Build orchestration ---------------------------------------------------

#[derive(Clone, Debug)]
pub enum BuildState {
    Running,
    Success,
    Failed(String),
}

#[derive(Clone, Debug)]
pub struct BuildSnapshot {
    pub state: BuildState,
    pub elapsed: Duration,
    /// Tail of captured output, oldest -> newest.
    pub lines: Vec<String>,
    /// Rough proxy for progress: how many "Compiling" lines we've seen.
    pub compiles: usize,
    #[allow(dead_code)]
    pub crate_name: String,
}

struct Shared {
    state: BuildState,
    started: Instant,
    lines: std::collections::VecDeque<String>,
    compiles: usize,
    crate_name: String,
}

/// A backgrounded `cargo build` whose output can be polled from the UI thread.
pub struct BuildJob {
    shared: Arc<Mutex<Shared>>,
    _join: Option<JoinHandle<()>>,
}

impl BuildJob {
    /// Start a backgrounded build. When `force` is true, `torch-sys`'s build
    /// artifacts are cleaned first so it re-detects the active libtorch — this
    /// is needed when a stale client was linked against a different/older
    /// libtorch than the one currently installed in the sandbox.
    pub fn start(crate_name: &str, force: bool) -> Self {
        let crate_name = crate_name.to_string();
        let shared = Arc::new(Mutex::new(Shared {
            state: BuildState::Running,
            started: Instant::now(),
            lines: std::collections::VecDeque::with_capacity(512),
            compiles: 0,
            crate_name: crate_name.clone(),
        }));

        let env = Env::detect();
        let shared_for_thread = shared.clone();
        let join = thread::spawn(move || {
            if force {
                push_line(&shared_for_thread, "forcing torch-sys rebuild (libtorch changed)".into());
                let mut clean = Command::new(&env.cargo);
                clean.arg("clean").arg("-p").arg("torch-sys");
                env.apply(&mut clean);
                clean.current_dir(config::repo_root());
                let _ = clean.output(); // best-effort; fast
            }

            let mut cmd = Command::new(&env.cargo);
            cmd.arg("build")
                .arg("--release")
                .arg("-p")
                .arg(&crate_name)
                .current_dir(config::repo_root())
                .stdout(Stdio::null())
                .stderr(Stdio::piped());
            env.apply(&mut cmd);

            let child = cmd.spawn();
            let mut child = match child {
                Ok(c) => c,
                Err(e) => {
                    push_line(&shared_for_thread, format!("failed to start cargo: {e}"));
                    set_state(&shared_for_thread, BuildState::Failed(e.to_string()));
                    return;
                }
            };
            if let Some(stderr) = child.stderr.take() {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    push_line(&shared_for_thread, line);
                }
            }
            let result = child.wait();
            let state = match result {
                Ok(status) if status.success() => BuildState::Success,
                Ok(status) => BuildState::Failed(format!("cargo exited with {status}")),
                Err(e) => BuildState::Failed(e.to_string()),
            };
            set_state(&shared_for_thread, state);
        });

        Self {
            shared,
            _join: Some(join),
        }
    }

    pub fn snapshot(&self) -> BuildSnapshot {
        let guard = self.shared.lock().unwrap();
        let tail: Vec<String> = guard
            .lines
            .iter()
            .rev()
            .take(16)
            .filter(|l| !l.trim().is_empty())
            .cloned()
            .collect();
        let tail: Vec<String> = tail.into_iter().rev().collect();
        BuildSnapshot {
            state: guard.state.clone(),
            elapsed: guard.started.elapsed(),
            compiles: guard.compiles,
            lines: tail,
            crate_name: guard.crate_name.clone(),
        }
    }
}

fn push_line(shared: &Arc<Mutex<Shared>>, line: String) {
    let mut g = shared.lock().unwrap();
    if line.starts_with("    Compiling ") {
        g.compiles += 1;
    }
    g.lines.push_back(line);
    if g.lines.len() > 512 {
        g.lines.pop_front();
    }
}

fn set_state(shared: &Arc<Mutex<Shared>>, state: BuildState) {
    let mut g = shared.lock().unwrap();
    // Don't overwrite a terminal state (e.g. a stray late line).
    if matches!(g.state, BuildState::Success | BuildState::Failed(_))
        && !matches!(state, BuildState::Success | BuildState::Failed(_))
    {
        return;
    }
    g.state = state;
}

/// Smoke-test the existing client binary against the active libtorch. The
/// dynamic linker resolves libtorch symbols at process start, so even
/// `--help` fails (e.g. with an "undefined symbol" error) when the binary was
/// linked against a different libtorch than the one currently installed.
pub fn client_runs() -> bool {
    let bin = config::client_bin();
    if !bin.exists() {
        return false;
    }
    let env = Env::detect();
    let mut cmd = Command::new(&bin);
    cmd.arg("--help");
    env.apply(&mut cmd);
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    matches!(cmd.output(), Ok(o) if o.status.success())
}

/// True if libtorch was (re)installed more recently than the client binary was
/// built — i.e. the binary is ABI-stale relative to the active torch. A rebuild
/// is needed even if the smoke check passes, since a 2.x-built binary running on
/// 2.y can load but then crash on ABI differences.
pub fn torch_changed_since_build() -> bool {
    let Ok(bin_meta) = std::fs::metadata(config::client_bin()) else {
        return false;
    };
    let Ok(bin_mod) = bin_meta.modified() else {
        return false;
    };
    let Some(torch_dir) = Env::detect().torch_lib_dirs.first().cloned() else {
        return false;
    };
    let torch_so = torch_dir.join("libtorch.so");
    let Ok(t_meta) = std::fs::metadata(&torch_so) else {
        return false;
    };
    matches!(t_meta.modified(), Ok(t) if t > bin_mod)
}

/// Replace this process with the training client.
pub fn exec_client(launch: &config::LaunchConfig) -> Result<()> {
    let bin = config::client_bin();
    if !bin.exists() {
        anyhow::bail!(
            "client binary not found at {}. The build screen should have produced it.",
            bin.display()
        );
    }

    let env = Env::detect();
    let mut cmd = Command::new(&bin);
    cmd.args(launch.client_args()).current_dir(config::repo_root());
    env.apply(&mut cmd);

    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        let err = cmd.exec();
        // exec only returns on failure.
        Err(anyhow::Error::from(err).context("exec training client"))
    }
    #[cfg(not(unix))]
    {
        let _ = cmd.status().context("run training client")?;
        Ok(())
    }
}
