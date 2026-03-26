//! Face alignment utilities
//!
//! This module provides face alignment operations:
//! - Aligning detected faces to canonical pose using affine transformations
//! - Pasting swapped faces back into original images

use crate::types::{AlignedFace, DetectedFace, FaceSwapError, Result};
use crate::utils::blur::gaussian_blur_2d;
use crate::utils::cv::erode_mask_optimized;
use crate::utils::transform::{estimate_affine_transform, invert_affine_transform, warp_affine};
use ndarray::{Array2, Array3, Array4};

// Standard landmarks for 112x112 alignment
// These are the target positions for the 5 facial landmarks (left eye, right eye, nose, left mouth, right mouth)
// when aligning a face to a 112x112 image for ArcFace recognition models
// https://github.com/deepinsight/insightface/blob/master/python-package/insightface/utils/face_align.py#L6
const LANDMARKS_112: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

// Standard landmarks for 128x128 alignment
// These are the target positions for the 5 facial landmarks when aligning a face to 128x128
// for face swapping models like inswapper_128
// Calculated as: arcface_dst * (128/128) + [8.0, 0] where arcface_dst is the base template
const LANDMARKS_128: [[f32; 2]; 5] = [
    [46.2946, 51.6963],
    [81.5318, 51.5014],
    [64.0252, 71.7366],
    [49.5493, 92.3655],
    [78.7299, 92.2041],
];

// Standard landmarks for 512x512 alignment
// These are the target positions for GFPGAN face enhancement
// Calculated as: LANDMARKS_128 * (512/128) = LANDMARKS_128 * 4
const LANDMARKS_512: [[f32; 2]; 5] = [
    [185.1784, 207.8852],
    [326.1272, 206.0056],
    [256.1008, 285.4464],
    [198.1972, 370.9620],
    [313.9196, 368.8164],
];

/// Align face to target size using affine transformation
///
/// # Arguments
/// * `img` - Source image as Array3<u8> (H, W, 3)
/// * `face` - Detected face with landmarks
/// * `target_size` - Target size (112, 128, or 512)
///
/// # Returns
/// AlignedFace containing face data, aligned image, and transformation matrix
pub fn align_face(img: &Array3<u8>, face: &DetectedFace, target_size: u32) -> Result<AlignedFace> {
    let target_landmarks = match target_size {
        112 => &LANDMARKS_112,
        128 => &LANDMARKS_128,
        512 => &LANDMARKS_512,
        _ => {
            return Err(FaceSwapError::InvalidInput(format!(
                "Unsupported target size: {}. Use 112, 128, or 512",
                target_size
            )))
        }
    };

    let transform = estimate_affine_transform(&face.landmarks, target_landmarks)?;

    // warp_affine expects aligned_coords -> original_coords
    // transform is: original_coords -> aligned_coords
    // So we need inverse? It is aligned_coords -> original_coords
    let inverse_for_warp = invert_affine_transform(&transform)?;
    let aligned = warp_affine(img, &inverse_for_warp, target_size as usize)?;

    let mut aligned_image = Array4::zeros((1, 3, target_size as usize, target_size as usize));
    for y in 0..target_size as usize {
        for x in 0..target_size as usize {
            aligned_image[[0, 0, y, x]] = aligned[[y, x, 0]] as f32 / 255.0;
            aligned_image[[0, 1, y, x]] = aligned[[y, x, 1]] as f32 / 255.0;
            aligned_image[[0, 2, y, x]] = aligned[[y, x, 2]] as f32 / 255.0;
        }
    }

    Ok(AlignedFace {
        face: face.clone(),
        aligned_image,
        transform,
    })
}

/// Paste swapped face back to original image with feathered mask blending
///
/// Creates a smooth blend between swapped face and original image using:
/// 1. White mask creation and inverse warping back to original coordinates
/// 2. Erosion to shrink mask inward from edges
/// 3. Gaussian blur for smooth feathering
/// 4. Alpha blending: result = mask * swapped + (1-mask) * original
///
/// This prevents hard rectangular boundaries visible in the output.
///
/// # Arguments
/// * `target_img` - Target image as Array3<u8> (H, W, 3)
/// * `swapped_face` - Swapped face as Array4<f32> (1, 3, size, size), normalized
/// * `align_transform` - Original alignment transformation matrix from align_face
/// * `bbox` - Face bounding box for region optimization
/// * `face_size` - Size of swapped face (128, 512)
///
/// # Returns
/// Target image with swapped face pasted using feathered mask
#[deprecated(
    since = "0.1.0",
    note = "Use paste_back_inplace instead for much better performance on high-resolution images"
)]
pub fn paste_back(
    target_img: &Array3<u8>,
    swapped_face: &Array4<f32>,
    align_transform: &Array2<f32>,
    _bbox: &crate::types::BBox,
    face_size: u32,
) -> Result<Array3<u8>> {
    let mut result = target_img.clone();
    #[allow(deprecated)]
    paste_back_inplace(&mut result, swapped_face, align_transform, face_size)?;
    Ok(result)
}

/// Paste swapped face back to original image in-place (ROI-optimized)
///
/// Optimized version that only operates on the face region instead of the full image.
/// For high-resolution images (e.g. 4K at 3840x2160), this is ~50-100x faster than
/// iterating over all 8.3M pixels. See `bench_roi_vs_fullimage` in optimization_bench.rs.
///
/// Pipeline within the face ROI:
/// 1. Compute face ROI from inverse affine transform of aligned face corners
/// 2. Warp mask + swapped face within ROI only (combined single pass)
/// 3. Erosion on ROI-sized mask to shrink edges
/// 4. Gaussian blur on ROI-sized mask for feathering
/// 5. Alpha blend directly into target_img (no full-image clone)
///
/// # Arguments
/// * `target_img` - Mutable target image as Array3<u8> (H, W, 3), modified in-place
/// * `swapped_face` - Swapped face as Array4<f32> (1, 3, size, size), normalized [0,1]
/// * `align_transform` - Alignment transformation matrix (original->aligned) from align_face
/// * `face_size` - Size of swapped face (128, 512)
pub fn paste_back_inplace(
    target_img: &mut Array3<u8>,
    swapped_face: &Array4<f32>,
    align_transform: &Array2<f32>,
    face_size: u32,
) -> Result<()> {
    let face_size_usize = face_size as usize;
    let (h, w, _) = target_img.dim();

    // Convert swapped face from normalized f32 to u8
    let swapped_u8 = Array3::from_shape_fn((face_size_usize, face_size_usize, 3), |(y, x, c)| {
        (swapped_face[[0, c, y, x]] * 255.0).clamp(0.0, 255.0) as u8
    });

    // Compute face ROI by inverse-transforming aligned face corners to target image space
    let inv = invert_affine_transform(align_transform)?;
    let corners: [(f32, f32); 4] = [
        (0.0, 0.0),
        (face_size as f32, 0.0),
        (0.0, face_size as f32),
        (face_size as f32, face_size as f32),
    ];

    let mut roi_min_x = f32::MAX;
    let mut roi_min_y = f32::MAX;
    let mut roi_max_x = f32::MIN;
    let mut roi_max_y = f32::MIN;
    for &(cx, cy) in &corners {
        let tx = inv[[0, 0]] * cx + inv[[0, 1]] * cy + inv[[0, 2]];
        let ty = inv[[1, 0]] * cx + inv[[1, 1]] * cy + inv[[1, 2]];
        roi_min_x = roi_min_x.min(tx);
        roi_min_y = roi_min_y.min(ty);
        roi_max_x = roi_max_x.max(tx);
        roi_max_y = roi_max_y.max(ty);
    }

    // Compute mask_size for erosion/blur kernel sizing
    let face_roi_w = (roi_max_x - roi_min_x).max(0.0) as usize;
    let face_roi_h = (roi_max_y - roi_min_y).max(0.0) as usize;
    let mask_size = ((face_roi_w * face_roi_h) as f32).sqrt() as usize;

    // Erosion kernel size (same formula as original paste_back)
    let erosion_k = (mask_size / 10).max(10);
    let erosion_k = if erosion_k % 2 == 0 {
        erosion_k + 1
    } else {
        erosion_k
    };

    // Blur kernel size (same formula as original paste_back)
    let blur_half = (mask_size / 20).max(5);
    let blur_k = 2 * blur_half + 1;

    // Add padding for erosion + blur kernels so edge effects don't reach the face
    let padding = (erosion_k / 2 + blur_k / 2 + 2) as f32;
    let roi_x0 = (roi_min_x - padding).max(0.0) as usize;
    let roi_y0 = (roi_min_y - padding).max(0.0) as usize;
    let roi_x1 = ((roi_max_x + padding).ceil() as usize + 1).min(w);
    let roi_y1 = ((roi_max_y + padding).ceil() as usize + 1).min(h);
    let roi_w = roi_x1 - roi_x0;
    let roi_h = roi_y1 - roi_y0;

    // Extract transform coefficients (original -> aligned mapping)
    let a = align_transform[[0, 0]];
    let b = align_transform[[0, 1]];
    let c = align_transform[[0, 2]];
    let d = align_transform[[1, 0]];
    let e = align_transform[[1, 1]];
    let f = align_transform[[1, 2]];

    // Warp mask and face within ROI only (combined single pass)
    // The white mask is uniform 255, so any pixel mapping inside the face bounds
    // produces mask=255 after threshold. This lets us skip the mask interpolation
    // and directly set mask=255 when inside bounds.
    let mut warped_mask = Array2::<u8>::zeros((roi_h, roi_w));
    let mut warped_face = Array3::<f32>::zeros((roi_h, roi_w, 3));

    for y in 0..roi_h {
        for x in 0..roi_w {
            let img_x = (roi_x0 + x) as f32;
            let img_y = (roi_y0 + y) as f32;
            let src_x = a * img_x + b * img_y + c;
            let src_y = d * img_x + e * img_y + f;

            if src_x >= 0.0
                && src_x < (face_size - 1) as f32
                && src_y >= 0.0
                && src_y < (face_size - 1) as f32
            {
                warped_mask[[y, x]] = 255;

                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(face_size_usize - 1);
                let y1 = (y0 + 1).min(face_size_usize - 1);
                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                for ch in 0..3 {
                    let v00 = swapped_u8[[y0, x0, ch]] as f32;
                    let v01 = swapped_u8[[y0, x1, ch]] as f32;
                    let v10 = swapped_u8[[y1, x0, ch]] as f32;
                    let v11 = swapped_u8[[y1, x1, ch]] as f32;
                    let v0 = v00 * (1.0 - dx) + v01 * dx;
                    let v1 = v10 * (1.0 - dx) + v11 * dx;
                    warped_face[[y, x, ch]] = v0 * (1.0 - dy) + v1 * dy;
                }
            }
        }
    }

    // Erosion on ROI-sized mask
    let eroded_mask = erode_mask_optimized(&warped_mask, erosion_k);

    // Gaussian blur on ROI-sized mask for feathering
    let eroded_mask_f32 = eroded_mask.mapv(|v| v as f32);
    let blurred_mask = gaussian_blur_2d(&eroded_mask_f32, blur_k);

    // Alpha blend directly into target_img within ROI (no full-image clone)
    for y in 0..roi_h {
        for x in 0..roi_w {
            let alpha = blurred_mask[[y, x]] / 255.0;
            if alpha > 0.0 {
                for ch in 0..3 {
                    let swapped_val = warped_face[[y, x, ch]];
                    let original_val = target_img[[roi_y0 + y, roi_x0 + x, ch]] as f32;
                    let blended = alpha * swapped_val + (1.0 - alpha) * original_val;
                    target_img[[roi_y0 + y, roi_x0 + x, ch]] = blended.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(())
}
