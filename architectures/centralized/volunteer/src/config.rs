//! Defaults, sandbox path layout, identity-key handling and the final argument
//! vector handed to the training client.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub const DEFAULT_RUN_ID: &str = "ds-v3-dense-250m-ufw";
pub const DEFAULT_SERVER_HOST: &str = "train.aethercompute.org";
pub const DEFAULT_SERVER_PORT: &str = "39405";
pub const DEFAULT_MICRO_BATCH: &str = "1";
pub const DEFAULT_SLOT: &str = "1";

pub const CLIENT_CRATE: &str = "psyche-centralized-client";
pub const CLIENT_BIN_NAME: &str = "psyche-centralized-client";

/// Repo root, resolved at runtime from this crate's compile-time manifest dir
/// (the volunteer crate lives at `<root>/architectures/centralized/volunteer`).
pub fn repo_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .ancestors()
        .nth(3)
        .map(Path::to_path_buf)
        .unwrap_or_else(|| manifest.clone())
}

/// Everything sandboxed lives under `<repo>/.aethercompute`.
pub fn sandbox_dir() -> PathBuf {
    repo_root().join(".aethercompute")
}

pub fn sandbox_cargo_home() -> PathBuf {
    sandbox_dir().join("cargo")
}

pub fn sandbox_rustup_home() -> PathBuf {
    sandbox_dir().join("rustup")
}

pub fn sandbox_venv() -> PathBuf {
    sandbox_dir().join("venv")
}

pub fn client_bin() -> PathBuf {
    repo_root()
        .join("target")
        .join("release")
        .join(CLIENT_BIN_NAME)
}

/// Per-slot working directory: identity key + logs.
pub fn slot_dir(slot: &str) -> PathBuf {
    let clean: String = slot
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') {
                c
            } else {
                '_'
            }
        })
        .collect();
    let clean = if clean.is_empty() {
        "1".to_string()
    } else {
        clean
    };
    sandbox_dir().join("clients").join(clean)
}

/// Ensure a 32-byte identity key exists. Returns `true` if it was just created.
pub fn ensure_identity_key(path: &Path) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).with_context(|| format!("mkdir {parent:?}"))?;
    }
    let mut bytes = [0u8; 32];
    read_urandom(&mut bytes).context("reading /dev/urandom for identity key")?;
    std::fs::write(path, bytes).with_context(|| format!("writing {path:?}"))?;
    Ok(true)
}

fn read_urandom(buf: &mut [u8]) -> Result<()> {
    use std::io::Read;
    let mut f = std::fs::File::open("/dev/urandom")?;
    f.read_exact(buf)?;
    Ok(())
}

/// Final, resolved configuration gathered by the TUI and handed to `exec`.
#[derive(Clone, Debug)]
pub struct LaunchConfig {
    pub run_id: String,
    pub server_addr: String, // host:port
    pub device: String,
    pub micro_batch_size: String,
    pub identity_key: PathBuf,
    pub log_file: PathBuf,
}

impl LaunchConfig {
    pub fn client_args(&self) -> Vec<String> {
        vec![
            "train".to_string(),
            "--run-id".to_string(),
            self.run_id.clone(),
            "--server-addr".to_string(),
            self.server_addr.clone(),
            "--device".to_string(),
            self.device.clone(),
            "--micro-batch-size".to_string(),
            self.micro_batch_size.clone(),
            "--identity-secret-key-path".to_string(),
            self.identity_key.to_string_lossy().into_owned(),
            "--logs".to_string(),
            "tui".to_string(),
            "--write-log".to_string(),
            self.log_file.to_string_lossy().into_owned(),
        ]
    }
}
