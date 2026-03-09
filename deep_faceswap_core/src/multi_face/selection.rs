//! Face crop saving and mapping orchestration

use crate::types::{DetectedFace, FaceCropInfo, FaceMapping, Result};
use crate::multi_face::prompt;
use ndarray::Array3;
use std::fs;

/// Save detected faces as crop images for visual inspection
///
/// Creates directory structure ./tmp/face_crops/{source,target}/
/// and saves each face as face_N.jpg where N is the face index.
///
/// # Arguments
/// * `faces` - Detected faces sorted by score (descending)
/// * `image` - Source image in HWC format (RGB, u8)
/// * `is_source` - true for source image, false for target image
///
/// # Returns
/// Vector of FaceCropInfo with paths to saved crops
pub fn save_face_crops(
    faces: &[DetectedFace],
    image: &Array3<u8>,
    is_source: bool,
) -> Result<Vec<FaceCropInfo>> {
    let subdir = if is_source { "source" } else { "target" };
    let crop_dir = format!("./tmp/face_crops/{}", subdir);

    fs::create_dir_all(&crop_dir)?;

    let mut crop_infos = Vec::new();

    for (idx, face) in faces.iter().enumerate() {
        let crop_path = format!("{}/face_{}.jpg", crop_dir, idx);

        // Extract face region with 20% padding
        let bbox = &face.bbox;
        let padding = 0.2;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;
        let pad_w = w * padding;
        let pad_h = h * padding;

        let x1 = (bbox.x1 - pad_w).max(0.0) as usize;
        let y1 = (bbox.y1 - pad_h).max(0.0) as usize;
        let x2 = (bbox.x2 + pad_w).min((image.shape()[1] - 1) as f32) as usize;
        let y2 = (bbox.y2 + pad_h).min((image.shape()[0] - 1) as f32) as usize;

        let crop_h = y2 - y1;
        let crop_w = x2 - x1;

        // Extract crop
        let mut crop = Array3::<u8>::zeros((crop_h, crop_w, 3));
        for y in 0..crop_h {
            for x in 0..crop_w {
                for c in 0..3 {
                    crop[[y, x, c]] = image[[y1 + y, x1 + x, c]];
                }
            }
        }

        // Convert to image and save
        let img_buffer = image::RgbImage::from_fn(crop_w as u32, crop_h as u32, |x, y| {
            image::Rgb([
                crop[[y as usize, x as usize, 0]],
                crop[[y as usize, x as usize, 1]],
                crop[[y as usize, x as usize, 2]],
            ])
        });

        img_buffer.save(&crop_path)?;

        crop_infos.push(FaceCropInfo {
            face: face.clone(),
            crop_path,
            index: idx,
        });
    }

    Ok(crop_infos)
}

/// Build face mappings based on number of detected faces
///
/// Orchestrates the interactive selection flow:
/// - (1, 1): Single mapping, no prompt
/// - (1, N): Prompt user to select target faces
/// - (N, 1): Prompt user to select source face
/// - (N, N): Prompt user for full mapping
///
/// # Arguments
/// * `source_faces` - Detected faces in source image
/// * `target_faces` - Detected faces in target image
/// * `source_img` - Source image for crop saving
/// * `target_img` - Target image for crop saving
///
/// # Returns
/// Vector of FaceMapping specifying source->target pairs
pub fn build_face_mappings(
    source_faces: &[DetectedFace],
    target_faces: &[DetectedFace],
    source_img: &Array3<u8>,
    target_img: &Array3<u8>,
) -> Result<Vec<FaceMapping>> {
    let n_source = source_faces.len();
    let n_target = target_faces.len();

    match (n_source, n_target) {
        (1, 1) => {
            // Simple case: one source, one target
            Ok(vec![FaceMapping {
                source_idx: 0,
                target_idx: 0,
            }])
        }
        (1, _) => {
            // One source, multiple targets
            let source_crops = save_face_crops(source_faces, source_img, true)?;
            let target_crops = save_face_crops(target_faces, target_img, false)?;

            let selected_targets = prompt::prompt_target_selection(&source_crops, &target_crops)?;

            Ok(selected_targets
                .into_iter()
                .map(|target_idx| FaceMapping {
                    source_idx: 0,
                    target_idx,
                })
                .collect())
        }
        (_, 1) => {
            // Multiple sources, one target
            let source_crops = save_face_crops(source_faces, source_img, true)?;
            let target_crops = save_face_crops(target_faces, target_img, false)?;

            let selected_source = prompt::prompt_source_selection(&source_crops, &target_crops)?;

            Ok(vec![FaceMapping {
                source_idx: selected_source,
                target_idx: 0,
            }])
        }
        _ => {
            // Multiple sources, multiple targets
            let source_crops = save_face_crops(source_faces, source_img, true)?;
            let target_crops = save_face_crops(target_faces, target_img, false)?;

            prompt::prompt_full_mapping(&source_crops, &target_crops)
        }
    }
}
