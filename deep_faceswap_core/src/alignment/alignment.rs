//! Face alignment utilities
//!
//! This module provides face alignment operations:
//! - Aligning detected faces to canonical pose using affine transformations
//! - Pasting swapped faces back into original images

use crate::types::{AlignedFace, DetectedFace, FaceSwapError, Result};
use crate::utils::transform::{estimate_affine_transform, warp_affine};
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
    let aligned = warp_affine(img, &transform, target_size as usize)?;

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

/// Paste swapped face back to original image
///
/// # Arguments
/// * `target_img` - Target image as Array3<u8> (H, W, 3)
/// * `swapped_face` - Swapped face as Array4<f32> (1, 3, size, size), normalized
/// * `align_transform` - Original alignment transformation matrix from align_face
/// * `bbox` - Face bounding box for region optimization
/// * `face_size` - Size of swapped face (128)
///
/// # Returns
/// Target image with swapped face pasted
pub fn paste_back(
    target_img: &Array3<u8>,
    swapped_face: &Array4<f32>,
    align_transform: &Array2<f32>,
    bbox: &crate::types::BBox,
    face_size: u32,
) -> Result<Array3<u8>> {
    let swapped_u8 =
        Array3::from_shape_fn((face_size as usize, face_size as usize, 3), |(y, x, c)| {
            (swapped_face[[0, c, y, x]] * 255.0).clamp(0.0, 255.0) as u8
        });

    let mut result = target_img.clone();

    let a = align_transform[[0, 0]];
    let b = align_transform[[0, 1]];
    let c = align_transform[[0, 2]];
    let d = align_transform[[1, 0]];
    let e = align_transform[[1, 1]];
    let f = align_transform[[1, 2]];

    let (h, w, _) = (result.shape()[0], result.shape()[1], result.shape()[2]);

    let x_min = bbox.x1.max(0.0) as usize;
    let y_min = bbox.y1.max(0.0) as usize;
    let x_max = (bbox.x2.min(w as f32) as usize).min(w);
    let y_max = (bbox.y2.min(h as f32) as usize).min(h);

    for y in y_min..y_max {
        for x in x_min..x_max {
            let src_x = a * x as f32 + b * y as f32 + c;
            let src_y = d * x as f32 + e * y as f32 + f;

            if src_x >= 0.0
                && src_x < (face_size - 1) as f32
                && src_y >= 0.0
                && src_y < (face_size - 1) as f32
            {
                let x0 = src_x.floor() as usize;
                let y0 = src_y.floor() as usize;
                let x1 = (x0 + 1).min(face_size as usize - 1);
                let y1 = (y0 + 1).min(face_size as usize - 1);

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

                    result[[y, x, ch]] = v.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }

    Ok(result)
}
