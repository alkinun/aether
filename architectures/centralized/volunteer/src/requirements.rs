//! Hardware requirements and the VRAM→micro-batch heuristic.
//!
//! The volunteer run is a 250M-param bf16 model (seq len 512) trained with the
//! compressed Distro optimizer, so optimizer/gradient state is small and peak
//! memory is dominated by activations — which scale with the micro-batch size.
//! That lets us recommend a micro-batch from VRAM alone, and gate the launch on
//! a minimum so volunteers don't OOM seconds into the run.
//!
//! All thresholds live here as constants so they're easy to tune in one place.

/// Minimum VRAM a single NVIDIA GPU must have to be allowed to train.
/// 4 GiB fits the 250M/bf16/Distro model comfortably at micro-batch 1.
pub const MIN_VRAM_MIB: u32 = 4 * 1024;

/// True when the given VRAM (in MiB) clears the minimum floor.
/// `None` (VRAM unknown — e.g. MPS/CPU/auto) is treated as "not gated": the
/// caller decides separately whether to allow it.
pub fn meets_minimum(vram_mib: Option<u32>) -> bool {
    match vram_mib {
        Some(mib) => mib >= MIN_VRAM_MIB,
        None => true,
    }
}

/// Recommend a micro-batch size from VRAM.
///
/// Returns `None` when the GPU is below the floor (caller must gate it out) or
/// when VRAM is unknown (caller falls back to its own default).
pub fn recommended_micro_batch(vram_mib: Option<u32>) -> Option<usize> {
    let mib = vram_mib?;
    if mib < MIN_VRAM_MIB {
        return None;
    }
    let batch = if mib < 8 * 1024 {
        1
    } else if mib < 16 * 1024 {
        2
    } else if mib < 24 * 1024 {
        4
    } else {
        8
    };
    Some(batch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_floor_is_blocked_and_not_recommended() {
        assert!(!meets_minimum(Some(2048)));
        assert_eq!(recommended_micro_batch(Some(2048)), None);
    }

    #[test]
    fn at_floor_passes_with_batch_one() {
        assert!(meets_minimum(Some(MIN_VRAM_MIB)));
        assert_eq!(recommended_micro_batch(Some(MIN_VRAM_MIB)), Some(1));
    }

    #[test]
    fn ladder_matches_vram() {
        assert_eq!(recommended_micro_batch(Some(6 * 1024)), Some(1));
        assert_eq!(recommended_micro_batch(Some(8 * 1024)), Some(2));
        assert_eq!(recommended_micro_batch(Some(12 * 1024)), Some(2));
        assert_eq!(recommended_micro_batch(Some(16 * 1024)), Some(4));
        assert_eq!(recommended_micro_batch(Some(24 * 1024)), Some(8));
        assert_eq!(recommended_micro_batch(Some(48 * 1024)), Some(8));
    }

    #[test]
    fn unknown_vram_is_not_gated_and_has_no_recommendation() {
        assert!(meets_minimum(None));
        assert_eq!(recommended_micro_batch(None), None);
    }
}
