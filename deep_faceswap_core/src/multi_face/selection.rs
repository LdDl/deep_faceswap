//! Face crop saving and mapping orchestration

use crate::multi_face::prompt;
use crate::types::{
    ClusterCropInfo, DetectedFace, FaceCropInfo, FaceMapping, Result, SourceFaceInfo,
};
use crate::video::ClusterInfo;
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
    save_face_crops_to(faces, image, is_source, "./tmp/face_crops")
}

/// Save detected faces as crop images to a custom directory
///
/// Like save_face_crops but with configurable base directory.
/// Creates {base_dir}/{source,target}/ subdirectory structure.
pub fn save_face_crops_to(
    faces: &[DetectedFace],
    image: &Array3<u8>,
    is_source: bool,
    base_dir: &str,
) -> Result<Vec<FaceCropInfo>> {
    let subdir = if is_source { "source" } else { "target" };
    let crop_dir = format!("{}/{}", base_dir, subdir);

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

/// Save source faces from multiple images as crop images for visual inspection
///
/// Creates directory ./tmp/face_crops/source/ and saves each face with filename
/// prefix to show which source image it came from (e.g., img1_face_0.jpg).
///
/// # Arguments
/// * `source_infos` - Source face info with origin metadata
/// * `source_images` - All loaded source images
///
/// # Returns
/// Vector of FaceCropInfo with paths to saved crops
pub fn save_face_crops_from_infos(
    source_infos: &[SourceFaceInfo],
    source_images: &[Array3<u8>],
) -> Result<Vec<FaceCropInfo>> {
    save_face_crops_from_infos_to(source_infos, source_images, "./tmp/face_crops/source")
}

/// Save source faces from multiple images to a custom directory
///
/// Like save_face_crops_from_infos but with configurable output directory.
pub fn save_face_crops_from_infos_to(
    source_infos: &[SourceFaceInfo],
    source_images: &[Array3<u8>],
    crop_dir: &str,
) -> Result<Vec<FaceCropInfo>> {
    fs::create_dir_all(crop_dir)?;

    let mut crop_infos = Vec::new();
    let mut face_counts: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();

    for (idx, info) in source_infos.iter().enumerate() {
        let local_face_idx = face_counts.entry(info.source_image_index).or_insert(0);

        let base_name = info
            .source_filename
            .trim_end_matches(".jpg")
            .trim_end_matches(".png")
            .trim_end_matches(".jpeg");

        let crop_path = format!("{}/{}_face_{}.jpg", crop_dir, base_name, local_face_idx);

        *local_face_idx += 1;

        let image = &source_images[info.source_image_index];
        let face = &info.face;
        let bbox = &face.bbox;

        // Extract face region with 20% padding
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
/// * `source_face_infos` - Source faces with origin metadata
/// * `target_faces` - Detected faces in target image
/// * `source_images` - All source images
/// * `target_img` - Target image for crop saving
///
/// # Returns
/// Vector of FaceMapping specifying source->target pairs
pub fn build_face_mappings(
    source_face_infos: &[SourceFaceInfo],
    target_faces: &[DetectedFace],
    source_images: &[Array3<u8>],
    target_img: &Array3<u8>,
) -> Result<Vec<FaceMapping>> {
    let n_source = source_face_infos.len();
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
            let source_crops = save_face_crops_from_infos(source_face_infos, source_images)?;
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
            let source_crops = save_face_crops_from_infos(source_face_infos, source_images)?;
            let target_crops = save_face_crops(target_faces, target_img, false)?;

            let selected_source = prompt::prompt_source_selection(&source_crops, &target_crops)?;

            Ok(vec![FaceMapping {
                source_idx: selected_source,
                target_idx: 0,
            }])
        }
        _ => {
            // Multiple sources, multiple targets
            let source_crops = save_face_crops_from_infos(source_face_infos, source_images)?;
            let target_crops = save_face_crops(target_faces, target_img, false)?;

            prompt::prompt_full_mapping(&source_crops, &target_crops)
        }
    }
}

/// Save cluster example faces as crop images for visual inspection
///
/// Creates directory ./tmp/face_crops/clusters/ and saves the best example
/// face from each cluster with metadata about frequency.
///
/// # Arguments
/// * `cluster_infos` - Cluster information with example faces
/// * `frame_paths` - All video frame paths to load example frames
///
/// # Returns
/// Vector of ClusterCropInfo with paths to saved crops
pub fn save_cluster_crops(
    cluster_infos: &[ClusterInfo],
    frame_paths: &[String],
) -> Result<Vec<ClusterCropInfo>> {
    save_cluster_crops_to(cluster_infos, frame_paths, "./tmp/face_crops/clusters")
}

/// Save cluster example faces to a custom directory
///
/// Like save_cluster_crops but with configurable output directory.
pub fn save_cluster_crops_to(
    cluster_infos: &[ClusterInfo],
    frame_paths: &[String],
    crop_dir: &str,
) -> Result<Vec<ClusterCropInfo>> {
    use crate::utils::{image as img_io, rgb};

    fs::create_dir_all(crop_dir)?;

    let mut crop_infos = Vec::new();

    for info in cluster_infos {
        let crop_path = format!("{}/cluster_{}.jpg", crop_dir, info.cluster_id);

        // Load the frame containing example face
        let frame_path = &frame_paths[info.example_frame_idx];
        let frame_img = img_io::load_image(frame_path)?;
        let frame_rgb = img_io::to_rgb8(&frame_img);
        let frame_array = rgb::rgb_to_array3(&frame_rgb);

        let face = &info.example_face;
        let bbox = &face.bbox;

        // Extract face region with 20% padding
        let padding = 0.2;
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;
        let pad_w = w * padding;
        let pad_h = h * padding;

        let x1 = (bbox.x1 - pad_w).max(0.0) as usize;
        let y1 = (bbox.y1 - pad_h).max(0.0) as usize;
        let x2 = (bbox.x2 + pad_w).min((frame_array.shape()[1] - 1) as f32) as usize;
        let y2 = (bbox.y2 + pad_h).min((frame_array.shape()[0] - 1) as f32) as usize;

        let crop_h = y2 - y1;
        let crop_w = x2 - x1;

        // Extract crop
        let mut crop = Array3::<u8>::zeros((crop_h, crop_w, 3));
        for y in 0..crop_h {
            for x in 0..crop_w {
                for c in 0..3 {
                    crop[[y, x, c]] = frame_array[[y1 + y, x1 + x, c]];
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

        crop_infos.push(ClusterCropInfo {
            face: info.example_face.clone(),
            crop_path,
            cluster_id: info.cluster_id,
            frame_count: info.frame_count,
        });
    }

    Ok(crop_infos)
}

/// Build cluster mappings for video multi-face swapping
///
/// Orchestrates the interactive selection flow for mapping source faces
/// to target clusters found in video frames.
///
/// # Arguments
/// * `source_face_infos` - Source faces with origin metadata
/// * `cluster_infos` - Cluster information from video analysis
/// * `source_images` - All source images
/// * `frame_paths` - All video frame paths
///
/// # Returns
/// Vector of ClusterMapping specifying source->cluster pairs
pub fn build_cluster_mappings(
    source_face_infos: &[SourceFaceInfo],
    cluster_infos: &[ClusterInfo],
    source_images: &[Array3<u8>],
    frame_paths: &[String],
) -> Result<Vec<crate::types::ClusterMapping>> {
    use crate::types::ClusterMapping;

    let n_source = source_face_infos.len();
    let n_clusters = cluster_infos.len();

    match (n_source, n_clusters) {
        (1, 1) => {
            // Simple case: one source, one cluster
            Ok(vec![ClusterMapping {
                source_idx: 0,
                cluster_id: 0,
            }])
        }
        (1, _) => {
            // One source, multiple clusters
            // Ask user which clusters to swap
            let source_crops = save_face_crops_from_infos(source_face_infos, source_images)?;
            let cluster_crops = save_cluster_crops(cluster_infos, frame_paths)?;

            let selected_clusters =
                prompt::prompt_cluster_selection(&source_crops, &cluster_crops)?;

            Ok(selected_clusters
                .into_iter()
                .map(|cluster_id| ClusterMapping {
                    source_idx: 0,
                    cluster_id,
                })
                .collect())
        }
        (_, 1) => {
            // Multiple sources, one cluster
            // Ask user which source to use
            let source_crops = save_face_crops_from_infos(source_face_infos, source_images)?;
            let cluster_crops = save_cluster_crops(cluster_infos, frame_paths)?;

            let mappings = prompt::prompt_cluster_mapping(&source_crops, &cluster_crops)?;

            // Validate that cluster_id is always 0
            for mapping in &mappings {
                if mapping.cluster_id != 0 {
                    return Err(crate::types::FaceSwapError::InvalidMapping(
                        "Only cluster ID 0 is valid when single cluster detected".to_string(),
                    ));
                }
            }

            if mappings.len() > 1 {
                return Err(crate::types::FaceSwapError::InvalidMapping(
                    "Only one mapping allowed when single cluster detected".to_string(),
                ));
            }

            Ok(mappings)
        }
        _ => {
            // Multiple sources, multiple clusters
            let source_crops = save_face_crops_from_infos(source_face_infos, source_images)?;
            let cluster_crops = save_cluster_crops(cluster_infos, frame_paths)?;

            prompt::prompt_cluster_mapping(&source_crops, &cluster_crops)
        }
    }
}
