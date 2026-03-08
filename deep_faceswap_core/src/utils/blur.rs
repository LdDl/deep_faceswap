//! Gaussian blur for 2D float arrays
//!
//! Provides separable Gaussian blur using two 1D passes (horizontal + vertical).
//! Used for softening mask edges in mouth mask blending.

use ndarray::Array2;

/// Generate 1D Gaussian kernel with OpenCV-compatible sigma.
///
/// When sigma is not specified, OpenCV's `getGaussianKernel` computes it as:
/// `sigma = 0.3 * ((ksize - 1) * 0.5 - 1) + 0.8`
/// See: https://github.com/opencv/opencv/blob/4.x/modules/imgproc/include/opencv2/imgproc.hpp#L1470
/// (search for `getGaussianKernel`)
///
/// The kernel is normalized so that all values sum to 1.0.
///
/// # Arguments
/// * `kernel_size` - Size of the kernel (must be odd and >= 3)
///
/// # Returns
/// Vector of kernel weights
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::blur::gaussian_kernel_1d;
///
/// let kernel = gaussian_kernel_1d(5);
/// assert_eq!(kernel.len(), 5);
///
/// let sum: f32 = kernel.iter().sum();
/// assert!((sum - 1.0).abs() < 1e-5);
/// ```
pub fn gaussian_kernel_1d(kernel_size: usize) -> Vec<f32> {
    let half = (kernel_size / 2) as f32;
    let sigma = 0.3 * (half - 1.0) + 0.8;
    let mut kernel = vec![0.0f32; kernel_size];
    let mut sum = 0.0;
    for i in 0..kernel_size {
        let x = i as f32 - half;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel[i] = val;
        sum += val;
    }
    for v in kernel.iter_mut() {
        *v /= sum;
    }
    kernel
}

/// Apply separable Gaussian blur to a 2D float array.
///
/// Uses two 1D passes (horizontal then vertical) for efficiency.
/// Border pixels use clamped (replicate) padding.
///
/// # Arguments
/// * `input` - 2D float array to blur
/// * `kernel_size` - Gaussian kernel size (must be odd and >= 3)
///
/// # Panics
/// Panics if `kernel_size` is less than 3 or even.
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use deep_faceswap_core::utils::blur::gaussian_blur_2d;
///
/// let mut mask = Array2::<f32>::zeros((100, 100));
/// // Fill a rectangle in the center
/// for y in 30..70 {
///     for x in 30..70 {
///         mask[[y, x]] = 1.0;
///     }
/// }
///
/// let blurred = gaussian_blur_2d(&mask, 15);
///
/// // Center is still close to 1.0
/// assert!(blurred[[50, 50]] > 0.99);
/// // Edge is smoothly blended
/// assert!(blurred[[30, 50]] > 0.0 && blurred[[30, 50]] < 1.0);
/// ```
pub fn gaussian_blur_2d(input: &Array2<f32>, kernel_size: usize) -> Array2<f32> {
    assert!(kernel_size >= 3 && kernel_size % 2 == 1);

    let (h, w) = (input.nrows(), input.ncols());
    let kernel = gaussian_kernel_1d(kernel_size);
    let half = kernel_size / 2;

    // Horizontal pass
    let mut temp = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in 0..kernel_size {
                let sx = (x as isize + k as isize - half as isize)
                    .max(0)
                    .min((w - 1) as isize) as usize;
                sum += input[[y, sx]] * kernel[k];
            }
            temp[[y, x]] = sum;
        }
    }

    // Vertical pass
    let mut output = Array2::<f32>::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0;
            for k in 0..kernel_size {
                let sy = (y as isize + k as isize - half as isize)
                    .max(0)
                    .min((h - 1) as isize) as usize;
                sum += temp[[sy, x]] * kernel[k];
            }
            output[[y, x]] = sum;
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let kernel = gaussian_kernel_1d(15);
        let sum: f32 = kernel.iter().sum();
        assert!((sum - 1.0).abs() < EPS);
    }

    #[test]
    fn test_blur_preserves_uniform() {
        let input = Array2::from_elem((50, 50), 0.5);
        let output = gaussian_blur_2d(&input, 15);
        for val in output.iter() {
            assert!((*val - 0.5).abs() < EPS);
        }
    }

    #[test]
    fn test_blur_smooths_spike() {
        let mut input = Array2::<f32>::zeros((21, 21));
        input[[10, 10]] = 1.0;
        let output = gaussian_blur_2d(&input, 5);
        // Center should be reduced
        assert!(output[[10, 10]] < 1.0);
        // Neighbors should be non-zero
        assert!(output[[10, 11]] > 0.0);
        assert!(output[[11, 10]] > 0.0);
    }
}
