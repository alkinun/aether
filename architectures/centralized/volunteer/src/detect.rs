//! Device discovery for the onboarding picker. Detects NVIDIA GPUs on Linux and
//! Apple Silicon on macOS. Everything is best-effort: if detection fails we just
//! fall back to "auto".

use std::process::Command;

#[derive(Clone, Debug)]
pub struct DeviceOption {
    /// Value passed to the client's `--device` flag (e.g. "cuda", "cuda:0").
    pub value: String,
    /// Human-readable label shown in the picker.
    pub label: String,
    /// Short tag rendered as a badge: GPU / CPU / NPU-ish.
    #[allow(dead_code)]
    pub tag: &'static str,
    /// Total VRAM in MiB when known (NVIDIA only). Used for the minimum-VRAM
    /// gate and the micro-batch heuristic; `None` for auto/cpu/mps.
    pub vram_mib: Option<u32>,
}

/// Build the list of device options, with the recommended one first.
pub fn detect_devices() -> Vec<DeviceOption> {
    let mut out = vec![DeviceOption {
        value: "auto".to_string(),
        label: "Auto — let Aether choose".to_string(),
        tag: "AUTO",
        vram_mib: None,
    }];

    let nvidia = detect_nvidia();
    if !nvidia.is_empty() {
        // "cuda" uses every visible GPU; attribute the *smallest* VRAM so the
        // minimum gate fails if any one of them is below the floor (it would
        // OOM during data-parallel sharding).
        let min_vram = nvidia.iter().filter_map(|(_, _, v)| *v).min();
        out.push(DeviceOption {
            value: "cuda".to_string(),
            label: "All visible NVIDIA GPUs".to_string(),
            tag: "CUDA",
            vram_mib: min_vram,
        });
        for (idx, name, vram) in &nvidia {
            let label = match vram {
                Some(mib) => format!("{name} · {mib} MiB (cuda:{idx})"),
                None => format!("{name} (cuda:{idx})"),
            };
            out.push(DeviceOption {
                value: format!("cuda:{idx}"),
                label,
                tag: "CUDA",
                vram_mib: *vram,
            });
        }
    }

    if cfg!(target_os = "macos")
        && is_apple_silicon() {
            out.push(DeviceOption {
                value: "mps".to_string(),
                label: "Apple Metal (MPS)".to_string(),
                tag: "MPS",
                vram_mib: None,
            });
        }

    out.push(DeviceOption {
        value: "cpu".to_string(),
        label: "CPU only".to_string(),
        tag: "CPU",
        vram_mib: None,
    });

    out
}

/// VRAM of the largest NVIDIA GPU on the host, if any. Used to pre-fill the
/// micro-batch field for the `auto` device path.
pub fn best_gpu_vram_mib() -> Option<u32> {
    detect_nvidia()
        .into_iter()
        .filter_map(|(_, _, v)| v)
        .max()
}

/// Returns `(index, name, vram_mib)` for each NVIDIA GPU, empty if none /
/// nvidia-smi missing. VRAM is `None` when the query ran but reported nothing.
fn detect_nvidia() -> Vec<(u32, String, Option<u32>)> {
    let out = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        Ok(o) if o.status.success() => o.stdout,
        _ => return Vec::new(),
    };
    let s = String::from_utf8_lossy(&out);
    s.lines()
        .filter_map(|l| {
            let mut it = l.split(',');
            let idx: u32 = it.next()?.trim().parse().ok()?;
            let name = it.next()?.trim().to_string();
            let vram = it.next().and_then(|v| v.trim().parse::<u32>().ok());
            Some((idx, name, vram))
        })
        .collect()
}

fn is_apple_silicon() -> bool {
    matches!(std::env::consts::ARCH, "aarch64" | "arm")
}

/// A short, friendly summary of the host for the welcome screen.
pub fn host_summary() -> String {
    let os = match std::env::consts::OS {
        "macos" => "macOS",
        "linux" => "Linux",
        other => other,
    };
    let arch = std::env::consts::ARCH;
    let gpus = detect_nvidia();
    let gpu_note = if !gpus.is_empty() {
        format!(" · {} NVIDIA GPU(s)", gpus.len())
    } else if os == "macOS" && is_apple_silicon() {
        " · Apple Silicon".to_string()
    } else {
        String::new()
    };
    format!("{os} ({arch}){gpu_note}")
}
