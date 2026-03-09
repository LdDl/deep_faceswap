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
pub fn paste_back(
    target_img: &Array3<u8>,
    swapped_face: &Array4<f32>,
    align_transform: &Array2<f32>,
    _bbox: &crate::types::BBox,
    face_size: u32,
) -> Result<Array3<u8>> {
    let face_size_usize = face_size as usize;

    // Convert swapped face from normalized f32 to u8
    let swapped_u8 = Array3::from_shape_fn((face_size_usize, face_size_usize, 3), |(y, x, c)| {
        (swapped_face[[0, c, y, x]] * 255.0).clamp(0.0, 255.0) as u8
    });

    let (h, w, _) = (
        target_img.shape()[0],
        target_img.shape()[1],
        target_img.shape()[2],
    );

    // Create white mask in aligned face space
    let white_mask = Array2::from_elem((face_size_usize, face_size_usize), 255.0f32);

    // align_transform is original->aligned (stored in AlignedFace from estimate_affine_transform)
    // For warping back, we need original->aligned to map target pixels to aligned space
    let a = align_transform[[0, 0]];
    let b = align_transform[[0, 1]];
    let c = align_transform[[0, 2]];
    let d = align_transform[[1, 0]];
    let e = align_transform[[1, 1]];
    let f = align_transform[[1, 2]];

    // Warp mask: for each pixel in target image, find source in aligned space
    let mut warped_mask = Array2::zeros((h, w));
    for y in 0..h {
        for x in 0..w {
            let src_x = a * x as f32 + b * y as f32 + c;
            let src_y = d * x as f32 + e * y as f32 + f;

            if src_x >= 0.0
                && src_x < (face_size - 1) as f32
                && src_y >= 0.0
                && src_y < (face_size - 1) as f32
            {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(face_size_usize - 1);
                let y1 = (y0 + 1).min(face_size_usize - 1);

                let dx = src_x - x0 as f32;
                let dy = src_y - y0 as f32;

                let v00 = white_mask[[y0, x0]];
                let v01 = white_mask[[y0, x1]];
                let v10 = white_mask[[y1, x0]];
                let v11 = white_mask[[y1, x1]];

                let v0 = v00 * (1.0 - dx) + v01 * dx;
                let v1 = v10 * (1.0 - dx) + v11 * dx;
                let v = v0 * (1.0 - dy) + v1 * dy;

                warped_mask[[y, x]] = v as u8;
            }
        }
    }

    // Threshold: white iimage[white>20] = 255
    for y in 0..h {
        for x in 0..w {
            if warped_mask[[y, x]] > 20 {
                warped_mask[[y, x]] = 255;
            } else {
                warped_mask[[y, x]] = 0;
            }
        }
    }

    // Warp swapped face using same transform
    let mut warped_face = Array3::zeros((h, w, 3));
    for y in 0..h {
        for x in 0..w {
            if warped_mask[[y, x]] == 0 {
                continue;
            }

            let src_x = a * x as f32 + b * y as f32 + c;
            let src_y = d * x as f32 + e * y as f32 + f;

            if src_x >= 0.0
                && src_x < (face_size - 1) as f32
                && src_y >= 0.0
                && src_y < (face_size - 1) as f32
            {
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
                    let v = v0 * (1.0 - dy) + v1 * dy;

                    warped_face[[y, x, ch]] = v;
                }
            }
        }
    }

    // Calculate mask size for erosion kernel
    let mask_h_inds: Vec<usize> = (0..h)
        .filter(|&y| (0..w).any(|x| warped_mask[[y, x]] == 255))
        .collect();
    let mask_w_inds: Vec<usize> = (0..w)
        .filter(|&x| (0..h).any(|y| warped_mask[[y, x]] == 255))
        .collect();

    let mask_h = if !mask_h_inds.is_empty() {
        mask_h_inds[mask_h_inds.len() - 1] - mask_h_inds[0]
    } else {
        0
    };
    let mask_w = if !mask_w_inds.is_empty() {
        mask_w_inds[mask_w_inds.len() - 1] - mask_w_inds[0]
    } else {
        0
    };

    let mask_size = ((mask_h * mask_w) as f32).sqrt() as usize;

    // Erosion kernel size: mask_size // 10, minimum 10
    let erosion_k = (mask_size / 10).max(10);
    let erosion_k = if erosion_k % 2 == 0 {
        erosion_k + 1
    } else {
        erosion_k
    };

    // Apply erosion to shrink mask inward
    let eroded_mask = erode_mask_optimized(&warped_mask, erosion_k);

    // Apply Gaussian blur
    // Kernel size: should be odd and positive
    let k = (mask_size / 20).max(5);
    let blur_k = 2 * k + 1;

    // Convert mask to f32 and apply Gaussian blur for feathering
    let eroded_mask_f32 = eroded_mask.mapv(|v| v as f32);
    let blurred_mask = gaussian_blur_2d(&eroded_mask_f32, blur_k);

    // Normalize mask to [0.0, 1.0] range for alpha blending
    let alpha_mask = blurred_mask.mapv(|v| v / 255.0);

    // Alpha blend: result = alpha_mask * warped_face + (1 - alpha_mask) * target_img
    let mut result = target_img.clone();
    for y in 0..h {
        for x in 0..w {
            let alpha = alpha_mask[[y, x]];
            for ch in 0..3 {
                let swapped_val = warped_face[[y, x, ch]];
                let original_val = target_img[[y, x, ch]] as f32;
                let blended = alpha * swapped_val + (1.0 - alpha) * original_val;
                result[[y, x, ch]] = blended.clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(result)
}
