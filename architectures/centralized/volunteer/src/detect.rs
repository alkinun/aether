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
}

/// Build the list of device options, with the recommended one first.
pub fn detect_devices() -> Vec<DeviceOption> {
    let mut out = vec![DeviceOption {
        value: "auto".to_string(),
        label: "Auto — let Aether choose".to_string(),
        tag: "AUTO",
    }];

    let nvidia = detect_nvidia();
    if !nvidia.is_empty() {
        out.push(DeviceOption {
            value: "cuda".to_string(),
            label: "All visible NVIDIA GPUs".to_string(),
            tag: "CUDA",
        });
        for (idx, name) in nvidia {
            out.push(DeviceOption {
                value: format!("cuda:{idx}"),
                label: format!("{name} (cuda:{idx})"),
                tag: "CUDA",
            });
        }
    }

    if cfg!(target_os = "macos")
        && is_apple_silicon() {
            out.push(DeviceOption {
                value: "mps".to_string(),
                label: "Apple Metal (MPS)".to_string(),
                tag: "MPS",
            });
        }

    out.push(DeviceOption {
        value: "cpu".to_string(),
        label: "CPU only".to_string(),
        tag: "CPU",
    });

    out
}

/// Returns `(index, name)` for each NVIDIA GPU, empty if none / nvidia-smi missing.
fn detect_nvidia() -> Vec<(u32, String)> {
    let out = match Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,name",
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
            Some((idx, name))
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
