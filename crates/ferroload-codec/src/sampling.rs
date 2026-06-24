//! Temporal frame-index selection for video (DESIGN §14.5: subsample *which*
//! frames to keep so we never decode-all-then-drop). Pure logic, fully tested;
//! used by the feature-gated ffmpeg decoder to choose frames before decoding.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sampling {
    /// Evenly spaced across the clip.
    Uniform,
    /// Contiguous run starting at the front.
    Dense,
}

/// Choose `num_frames` indices out of `total` according to `sampling`.
/// Always returns indices in `[0, total)`, ascending, length `min(num_frames, total)`.
pub fn frame_indices(total: usize, num_frames: usize, sampling: Sampling) -> Vec<usize> {
    if total == 0 || num_frames == 0 {
        return Vec::new();
    }
    let k = num_frames.min(total);
    match sampling {
        Sampling::Dense => (0..k).collect(),
        Sampling::Uniform => {
            if k == 1 {
                return vec![total / 2];
            }
            // evenly spaced incl. endpoints: round(i*(total-1)/(k-1))
            (0..k)
                .map(|i| ((i * (total - 1)) as f64 / (k - 1) as f64).round() as usize)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_spans_endpoints() {
        assert_eq!(frame_indices(10, 4, Sampling::Uniform), vec![0, 3, 6, 9]);
        assert_eq!(frame_indices(100, 1, Sampling::Uniform), vec![50]);
    }

    #[test]
    fn dense_is_prefix() {
        assert_eq!(frame_indices(10, 4, Sampling::Dense), vec![0, 1, 2, 3]);
    }

    #[test]
    fn clamps_when_fewer_frames_than_requested() {
        assert_eq!(frame_indices(3, 8, Sampling::Uniform), vec![0, 1, 2]);
        assert_eq!(frame_indices(0, 8, Sampling::Uniform), Vec::<usize>::new());
    }

    #[test]
    fn indices_in_range_and_sorted() {
        for total in [1usize, 5, 16, 257] {
            for nf in [1usize, 4, 16] {
                let idx = frame_indices(total, nf, Sampling::Uniform);
                assert!(idx.iter().all(|&i| i < total));
                assert!(idx.windows(2).all(|w| w[0] <= w[1]));
            }
        }
    }
}
