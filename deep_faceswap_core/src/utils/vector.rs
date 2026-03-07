//! Vector operations

/// L2 normalize a vector in-place
///
/// Normalizes the vector so that its L2 norm (Euclidean length) equals 1.
/// This is required for embeddings to be comparable via cosine similarity.
///
/// # Arguments
/// * `vec` - Vector to normalize in-place
///
/// # Algorithm
/// For vector v, computes: v = v / ||v||_{2} where ||v||_{2} = sqrt(sum(v_i^{2}))
///
/// # Example
/// ```rust
/// use deep_faceswap_core::utils::vector::l2_normalize;
///
/// let mut vec = vec![3.0, 4.0];
/// l2_normalize(&mut vec);
/// let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
/// assert!((norm - 1.0).abs() < 1e-6);
/// ```
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn test_l2_normalize_unit_vector() {
        let mut vec = vec![1.0, 0.0, 0.0];
        l2_normalize(&mut vec);

        assert!((vec[0] - 1.0).abs() < EPS);
        assert!((vec[1] - 0.0).abs() < EPS);
        assert!((vec[2] - 0.0).abs() < EPS);
    }

    #[test]
    fn test_l2_normalize_regular_vector() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize(&mut vec);

        let expected_norm = (3.0_f32 * 3.0 + 4.0 * 4.0).sqrt();
        assert!((vec[0] - 3.0 / expected_norm).abs() < EPS);
        assert!((vec[1] - 4.0 / expected_norm).abs() < EPS);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < EPS, "Normalized vector should have norm 1.0");
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut vec);

        assert_eq!(vec[0], 0.0);
        assert_eq!(vec[1], 0.0);
        assert_eq!(vec[2], 0.0);
    }

    #[test]
    fn test_l2_normalize_negative_values() {
        let mut vec = vec![-3.0, 4.0];
        l2_normalize(&mut vec);

        let expected_norm = (3.0_f32 * 3.0 + 4.0 * 4.0).sqrt();
        assert!((vec[0] - (-3.0) / expected_norm).abs() < EPS);
        assert!((vec[1] - 4.0 / expected_norm).abs() < EPS);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < EPS);
    }

    #[test]
    fn test_l2_normalize_large_vector() {
        let mut vec: Vec<f32> = (0..512).map(|i| (i as f32) / 512.0).collect();
        l2_normalize(&mut vec);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < EPS, "Normalized 512-d vector should have norm 1.0");
    }
}
