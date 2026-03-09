//! Affine transformation utilities

use crate::types::{FaceSwapError, Result};
use ndarray::{Array2, Array3};

const AFFINE_EPS: f32 = 1e-10;

/// Estimate similarity transformation matrix from source to target landmarks
///
/// Computes a similarity transformation (uniform scale + rotation + translation) that
/// best maps source landmarks to target landmarks using least squares fitting.
/// Similarity transforms preserve angles and have no shear distortion.
///
/// # Arguments
/// * `src_landmarks` - 5 source landmark points (e.g., detected face keypoints)
/// * `dst_landmarks` - 5 target landmark points (e.g., canonical face template)
///
/// # Returns
/// 2x3 transformation matrix [[a, -b, tx], [b, a, ty]] where:
/// - a = scale * cos(angle)
/// - b = scale * sin(angle)
/// - tx, ty = translation offsets
///
/// # Algorithm
/// Uses closed-form solution for similarity transform estimation via least squares
pub fn estimate_affine_transform(
    src_landmarks: &[[f32; 2]; 5],
    dst_landmarks: &[[f32; 2]; 5],
) -> Result<Array2<f32>> {
    let n = 5.0;

    let mut src_mean = [0.0f32; 2];
    let mut dst_mean = [0.0f32; 2];

    for i in 0..5 {
        src_mean[0] += src_landmarks[i][0];
        src_mean[1] += src_landmarks[i][1];
        dst_mean[0] += dst_landmarks[i][0];
        dst_mean[1] += dst_landmarks[i][1];
    }

    src_mean[0] /= n;
    src_mean[1] /= n;
    dst_mean[0] /= n;
    dst_mean[1] /= n;

    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xu_plus_yv = 0.0;
    let mut sum_xv_minus_yu = 0.0;

    for i in 0..5 {
        let x = src_landmarks[i][0] - src_mean[0];
        let y = src_landmarks[i][1] - src_mean[1];
        let u = dst_landmarks[i][0] - dst_mean[0];
        let v = dst_landmarks[i][1] - dst_mean[1];

        sum_xx += x * x;
        sum_yy += y * y;
        sum_xu_plus_yv += x * u + y * v;
        sum_xv_minus_yu += x * v - y * u;
    }

    let src_norm_sq = sum_xx + sum_yy;

    if src_norm_sq < AFFINE_EPS {
        return Err(FaceSwapError::ProcessingError(
            "Cannot estimate similarity transform: source points are degenerate".to_string(),
        ));
    }

    let a = sum_xu_plus_yv / src_norm_sq;
    let b = sum_xv_minus_yu / src_norm_sq;

    let tx = dst_mean[0] - (a * src_mean[0] - b * src_mean[1]);
    let ty = dst_mean[1] - (b * src_mean[0] + a * src_mean[1]);

    let transform = Array2::from_shape_vec((2, 3), vec![a, -b, tx, b, a, ty])
        .map_err(|e| FaceSwapError::ProcessingError(format!("{}", e)))?;

    Ok(transform)
}

/// Invert affine transformation matrix
///
/// Computes the inverse of a 2x3 affine transformation matrix.
/// The inverse maps from transformed coordinates back to original coordinates.
///
/// # Arguments
/// * `transform` - 2x3 affine matrix [[a, b, c], [d, e, f]]
///
/// # Returns
/// Inverse transformation matrix
///
/// # Errors
/// Returns error if the transformation is singular (determinant near zero)
///
pub fn invert_affine_transform(transform: &Array2<f32>) -> Result<Array2<f32>> {
    let a = transform[[0, 0]];
    let b = transform[[0, 1]];
    let c = transform[[0, 2]];
    let d = transform[[1, 0]];
    let e = transform[[1, 1]];
    let f = transform[[1, 2]];

    let det = a * e - b * d;
    if det.abs() < AFFINE_EPS {
        return Err(FaceSwapError::ProcessingError(
            "Affine transform is singular, cannot invert".to_string(),
        ));
    }

    let inv_a = e / det;
    let inv_b = -b / det;
    let inv_c = (b * f - e * c) / det;
    let inv_d = -d / det;
    let inv_e = a / det;
    let inv_f = (d * c - a * f) / det;

    let inverted = Array2::from_shape_vec((2, 3), vec![inv_a, inv_b, inv_c, inv_d, inv_e, inv_f])
        .map_err(|e| FaceSwapError::ProcessingError(format!("{}", e)))?;

    Ok(inverted)
}

/// Apply affine transformation to image
///
/// Warps an image using an affine transformation matrix with bilinear interpolation.
/// Matches cv2.warpAffine behavior: for each output pixel (x,y), computes
/// src = transform*(x,y) and samples from input image.
///
/// # Arguments
/// * `img` - Input image as HWC array (H, W, 3)
/// * `transform` - 2x3 affine transformation matrix mapping output_coords -> input_coords
/// * `output_size` - Size of output square image (e.g., 112 or 128)
///
/// # Returns
/// Transformed image as HWC array (output_size, output_size, 3)
///
/// # Note
/// For aligning face: pass M where M maps aligned_coords -> original_coords
/// For pasting back: pass IM where IM maps original_coords -> aligned_coords
pub fn warp_affine(
    img: &Array3<u8>,
    transform: &Array2<f32>,
    output_size: usize,
) -> Result<Array3<u8>> {
    let mut result = Array3::zeros((output_size, output_size, 3));

    let a = transform[[0, 0]];
    let b = transform[[0, 1]];
    let c = transform[[0, 2]];
    let d = transform[[1, 0]];
    let e = transform[[1, 1]];
    let f = transform[[1, 2]];

    let (h, w, _) = (img.shape()[0], img.shape()[1], img.shape()[2]);

    for y in 0..output_size {
        for x in 0..output_size {
            let src_x = a * x as f32 + b * y as f32 + c;
            let src_y = d * x as f32 + e * y as f32 + f;

            if src_x >= 0.0 && src_x < (w - 1) as f32 && src_y >= 0.0 && src_y < (h - 1) as f32 {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(w - 1);
                let y1 = (y0 + 1).min(h - 1);

                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                for c in 0..3 {
                    let v00 = img[[y0, x0, c]] as f32;
                    let v01 = img[[y0, x1, c]] as f32;
                    let v10 = img[[y1, x0, c]] as f32;
                    let v11 = img[[y1, x1, c]] as f32;

                    let v0 = v00 * (1.0 - dx) + v01 * dx;
                    let v1 = v10 * (1.0 - dx) + v11 * dx;
                    let v = v0 * (1.0 - dy) + v1 * dy;

                    result[[y, x, c]] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(result)
}

/// Apply affine transformation to a single-channel mask
///
/// Warps a mask using an affine transformation matrix with bilinear interpolation.
/// This is used to transform face masks back to the original image coordinates.
/// https://docs.opencv.org/4.x/da/d54/group__imgproc__transform.html#ga0203d9ee5fcd28d40dbc4a1ea4451983
///
/// NOTE: Uses FORWARD mapping like cv2.warpAffine!
///
/// # Arguments
/// * `mask` - Input mask as 2D array (H, W) with values 0-255
/// * `forward_transform` - 2x3 FORWARD affine transformation matrix (original->aligned)
/// * `output_h` - Height of output image
/// * `output_w` - Width of output image
///
/// # Returns
/// Transformed mask as 2D array (output_h, output_w)
pub fn warp_affine_mask(
    mask: &Array2<u8>,
    forward_transform: &Array2<f32>,
    output_h: usize,
    output_w: usize,
) -> Result<Array2<u8>> {
    let mut result = Array2::zeros((output_h, output_w));

    // Use forward transform and apply forward mapping
    let a = forward_transform[[0, 0]];
    let b = forward_transform[[0, 1]];
    let c = forward_transform[[0, 2]];
    let d = forward_transform[[1, 0]];
    let e = forward_transform[[1, 1]];
    let f = forward_transform[[1, 2]];

    let (h, w) = (mask.nrows(), mask.ncols());

    // FORWARD mapping: iterate over INPUT pixels and map to OUTPUT
    for src_y in 0..h {
        for src_x in 0..w {
            let dst_x = a * src_x as f32 + b * src_y as f32 + c;
            let dst_y = d * src_x as f32 + e * src_y as f32 + f;

            if dst_x >= 0.0
                && dst_x < (output_w - 1) as f32
                && dst_y >= 0.0
                && dst_y < (output_h - 1) as f32
            {
                let x0 = dst_x.floor() as usize;
                let y0 = dst_y.floor() as usize;
                let x1 = (x0 + 1).min(output_w - 1);
                let y1 = (y0 + 1).min(output_h - 1);

                let dx = dst_x - x0 as f32;
                let dy = dst_y - y0 as f32;

                let value = mask[[src_y, src_x]] as f32;

                // Bilinear splatting
                result[[y0, x0]] = (result[[y0, x0]] as f32 + value * (1.0 - dx) * (1.0 - dy))
                    .clamp(0.0, 255.0) as u8;
                result[[y0, x1]] =
                    (result[[y0, x1]] as f32 + value * dx * (1.0 - dy)).clamp(0.0, 255.0) as u8;
                result[[y1, x0]] =
                    (result[[y1, x0]] as f32 + value * (1.0 - dx) * dy).clamp(0.0, 255.0) as u8;
                result[[y1, x1]] =
                    (result[[y1, x1]] as f32 + value * dx * dy).clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    #[test]
    fn test_estimate_affine_transform_identity() {
        let landmarks = [
            [50.0, 50.0],
            [100.0, 50.0],
            [75.0, 75.0],
            [60.0, 100.0],
            [90.0, 100.0],
        ];

        let transform = estimate_affine_transform(&landmarks, &landmarks).unwrap();

        assert!((transform[[0, 0]] - 1.0).abs() < EPS, "a should be 1.0");
        assert!((transform[[0, 1]] - 0.0).abs() < EPS, "b should be 0.0");
        assert!((transform[[0, 2]] - 0.0).abs() < EPS, "tx should be 0.0");
        assert!((transform[[1, 0]] - 0.0).abs() < EPS, "b should be 0.0");
        assert!((transform[[1, 1]] - 1.0).abs() < EPS, "a should be 1.0");
        assert!((transform[[1, 2]] - 0.0).abs() < EPS, "ty should be 0.0");
    }

    #[test]
    fn test_estimate_affine_transform_translation() {
        let src = [
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 5.0],
            [2.0, 10.0],
            [8.0, 10.0],
        ];

        let mut dst = src;
        for point in &mut dst {
            point[0] += 20.0;
            point[1] += 30.0;
        }

        let transform = estimate_affine_transform(&src, &dst).unwrap();

        assert!((transform[[0, 0]] - 1.0).abs() < EPS, "a should be 1.0");
        assert!((transform[[0, 1]] - 0.0).abs() < EPS, "b should be 0.0");
        assert!((transform[[0, 2]] - 20.0).abs() < EPS, "tx should be 20.0");
        assert!((transform[[1, 0]] - 0.0).abs() < EPS, "b should be 0.0");
        assert!((transform[[1, 1]] - 1.0).abs() < EPS, "a should be 1.0");
        assert!((transform[[1, 2]] - 30.0).abs() < EPS, "ty should be 30.0");
    }

    #[test]
    fn test_estimate_affine_transform_scale() {
        let src = [
            [0.0, 0.0],
            [10.0, 0.0],
            [5.0, 5.0],
            [2.0, 10.0],
            [8.0, 10.0],
        ];

        let mut dst = src;
        for point in &mut dst {
            point[0] *= 2.0;
            point[1] *= 2.0;
        }

        let transform = estimate_affine_transform(&src, &dst).unwrap();

        assert!((transform[[0, 0]] - 2.0).abs() < EPS, "a should be 2.0");
        assert!((transform[[0, 1]] - 0.0).abs() < EPS, "b should be 0.0");
        assert!((transform[[1, 0]] - 0.0).abs() < EPS, "b should be 0.0");
        assert!((transform[[1, 1]] - 2.0).abs() < EPS, "a should be 2.0");
    }

    #[test]
    fn test_invert_affine_transform_identity() {
        let identity = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let inverted = invert_affine_transform(&identity).unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (inverted[[i, j]] - identity[[i, j]]).abs() < EPS,
                    "Identity inverse should be identity"
                );
            }
        }
    }

    #[test]
    fn test_invert_affine_transform_translation() {
        let translate =
            Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 10.0, 0.0, 1.0, 20.0]).unwrap();
        let inverted = invert_affine_transform(&translate).unwrap();

        assert!((inverted[[0, 0]] - 1.0).abs() < EPS);
        assert!((inverted[[0, 1]] - 0.0).abs() < EPS);
        assert!((inverted[[0, 2]] - (-10.0)).abs() < EPS);
        assert!((inverted[[1, 0]] - 0.0).abs() < EPS);
        assert!((inverted[[1, 1]] - 1.0).abs() < EPS);
        assert!((inverted[[1, 2]] - (-20.0)).abs() < EPS);
    }

    #[test]
    fn test_invert_affine_transform_roundtrip() {
        let original =
            Array2::from_shape_vec((2, 3), vec![2.0, -1.0, 5.0, 1.0, 2.0, -3.0]).unwrap();
        let inverted = invert_affine_transform(&original).unwrap();
        let roundtrip = invert_affine_transform(&inverted).unwrap();

        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (roundtrip[[i, j]] - original[[i, j]]).abs() < EPS,
                    "Roundtrip should return to original"
                );
            }
        }
    }

    #[test]
    fn test_estimate_affine_transform_degenerate() {
        let degenerate = [[0.0, 0.0]; 5];
        let target = [
            [50.0, 50.0],
            [100.0, 50.0],
            [75.0, 75.0],
            [60.0, 100.0],
            [90.0, 100.0],
        ];

        let result = estimate_affine_transform(&degenerate, &target);
        assert!(result.is_err(), "Degenerate points should return error");
    }

    #[test]
    fn test_invert_affine_transform_singular() {
        let singular = Array2::from_shape_vec((2, 3), vec![1.0, 1.0, 0.0, 1.0, 1.0, 0.0]).unwrap();
        let result = invert_affine_transform(&singular);
        assert!(result.is_err(), "Singular matrix should return error");
    }

    #[test]
    fn test_warp_affine_identity() {
        let mut img = Array3::<u8>::zeros((10, 10, 3));
        img[[5, 5, 0]] = 100;
        img[[5, 5, 1]] = 150;
        img[[5, 5, 2]] = 200;

        let identity = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let warped = warp_affine(&img, &identity, 10).unwrap();

        assert_eq!(warped[[5, 5, 0]], 100, "Identity should preserve pixels");
        assert_eq!(warped[[5, 5, 1]], 150, "Identity should preserve pixels");
        assert_eq!(warped[[5, 5, 2]], 200, "Identity should preserve pixels");
    }

    #[test]
    fn test_warp_affine_output_shape() {
        let img = Array3::<u8>::zeros((10, 10, 3));
        let identity = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();

        let warped = warp_affine(&img, &identity, 15).unwrap();
        assert_eq!(
            warped.shape(),
            &[15, 15, 3],
            "Output should match requested size"
        );
    }

    #[test]
    fn test_warp_affine_non_crash() {
        let mut img = Array3::<u8>::zeros((10, 10, 3));
        img[[5, 5, 0]] = 255;

        let transform = Array2::from_shape_vec((2, 3), vec![0.5, 0.0, 2.0, 0.0, 0.5, 3.0]).unwrap();
        let result = warp_affine(&img, &transform, 10);

        assert!(result.is_ok(), "Warp should not crash with valid transform");
    }

    #[test]
    fn test_warp_affine_mask_identity() {
        let mask = Array2::from_shape_fn((10, 10), |(y, x)| {
            if y >= 3 && y < 7 && x >= 3 && x < 7 {
                255u8
            } else {
                0u8
            }
        });

        let identity = Array2::from_shape_vec((2, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).unwrap();
        let warped = warp_affine_mask(&mask, &identity, 10, 10).unwrap();

        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(
                    warped[[y, x]],
                    mask[[y, x]],
                    "Identity should preserve mask"
                );
            }
        }
    }

    #[test]
    fn test_warp_affine_mask_scale() {
        let mut mask = Array2::<u8>::zeros((20, 20));
        for y in 8..12 {
            for x in 8..12 {
                mask[[y, x]] = 255;
            }
        }

        let scale = Array2::from_shape_vec((2, 3), vec![0.5, 0.0, 0.0, 0.0, 0.5, 0.0]).unwrap();
        let warped = warp_affine_mask(&mask, &scale, 10, 10).unwrap();

        assert!(
            warped[[5, 5]] > 0,
            "Center of scaled mask should have value"
        );
    }
}
