//! Main face swap operation

use crate::alignment;
use crate::detection::FaceDetector;
use crate::enhancer::FaceEnhancer;
use crate::landmark::LandmarkDetector;
use crate::mouth_mask;
use crate::multi_face;
use crate::recognition::FaceRecognizer;
use crate::swapper::FaceSwapper;
use crate::types::{DetectedFace, FaceSwapError, Result, SourceFaceInfo};
use crate::utils::image as img_io;
use crate::utils::rgb;
use crate::verbose::{EVENT_ALIGN_FACE, EVENT_COMPLETE, EVENT_FACE_DETECTED, EVENT_PASTE_BACK};
use crate::{log_additional, log_main};
use ndarray::Array3;

/// Swap a single source face into a single target face
///
/// # Arguments
/// * `target_image` - Mutable target image (HWC, RGB, u8)
/// * `target_face` - Detected face in target image
/// * `source_image` - Source image for face alignment (HWC, RGB, u8)
/// * `source_embedding` - Pre-extracted source face embedding (512D)
/// * `swapper` - Face swapper model
/// * `enhancer` - Optional face enhancer model
/// * `landmark_detector` - Optional 106-point landmark detector
/// * `use_mouth_mask` - Whether to apply mouth mask
///
/// # Returns
/// Modified target_image with swapped face
fn swap_single_pair(
    target_image: &mut Array3<u8>,
    target_face: &DetectedFace,
    _source_image: &Array3<u8>,
    source_embedding: &[f32],
    swapper: &mut FaceSwapper,
    enhancer: &mut Option<FaceEnhancer>,
    landmark_detector: &mut Option<LandmarkDetector>,
    use_mouth_mask: bool,
) -> Result<()> {
    // Detect 106 landmarks on target face before swap (needed for mouth mask)
    let mouth_mask_data = if use_mouth_mask {
        if let Some(ref mut lm_detector) = landmark_detector {
            log_additional!("mouth_mask", "Detecting 106 landmarks on target face");
            let landmarks = lm_detector.detect(target_image, target_face)?;
            let (frame_h, frame_w, _) = target_image.dim();
            let data = mouth_mask::create_mouth_mask(target_image, &landmarks, frame_h, frame_w)?;
            Some(data)
        } else {
            None
        }
    } else {
        None
    };

    log_additional!(EVENT_ALIGN_FACE, "Aligning target face");
    let target_aligned = alignment::align_face(target_image, target_face, 128)?;

    let swapped_face = swapper.swap(&target_aligned.aligned_image, source_embedding)?;

    log_additional!(EVENT_PASTE_BACK, "Pasting swapped face back");
    let mut result = alignment::paste_back(
        target_image,
        &swapped_face,
        &target_aligned.transform,
        &target_aligned.face.bbox,
        128,
    )?;

    // Apply mouth mask after swap but before enhancement
    if let Some(ref data) = mouth_mask_data {
        log_additional!("mouth_mask", "Applying mouth mask");
        mouth_mask::apply_mouth_mask(&mut result, data);
    }

    // Enhance face if enhancer is provided
    if let Some(ref mut enh) = enhancer {
        log_additional!("enhance_face", "Enhancing face at original resolution");

        // Align target face from result image to 512x512 for enhancement
        let target_aligned_512 = alignment::align_face(&result, target_face, 512)?;

        // Enhance the 512x512 aligned face
        let enhanced_512 = enh.enhance(&target_aligned_512.aligned_image)?;

        // Paste enhanced face back into result
        result = alignment::paste_back(
            &result,
            &enhanced_512,
            &target_aligned_512.transform,
            &target_aligned_512.face.bbox,
            512,
        )?;
    }

    *target_image = result;
    Ok(())
}

/// Swap multiple faces in target image based on interactive face mapping
///
/// # Arguments
/// * `source_faces` - All detected faces in source image
/// * `target_faces` - All detected faces in target image
/// * `source_image` - Source image (HWC, RGB, u8)
/// * `target_image` - Mutable target image (HWC, RGB, u8)
/// * `recognizer` - Face recognizer for embedding extraction
/// * `swapper` - Face swapper model
/// * `enhancer` - Optional face enhancer model
/// * `landmark_detector` - Optional 106-point landmark detector
/// * `use_mouth_mask` - Whether to apply mouth mask
///
/// # Returns
/// Modified target_image with all swapped faces
fn swap_multiple_faces(
    source_face_infos: &[SourceFaceInfo],
    target_faces: &[DetectedFace],
    source_images: &[Array3<u8>],
    target_image: &mut Array3<u8>,
    recognizer: &mut FaceRecognizer,
    swapper: &mut FaceSwapper,
    enhancer: &mut Option<FaceEnhancer>,
    landmark_detector: &mut Option<LandmarkDetector>,
    use_mouth_mask: bool,
) -> Result<()> {
    // Build face mappings via interactive prompts
    let mappings = multi_face::build_face_mappings(
        source_face_infos,
        target_faces,
        source_images,
        target_image,
    )?;

    log_main!(
        "multi_face",
        "Processing face mappings",
        count = mappings.len()
    );

    // Extract embeddings for all source faces upfront
    let mut source_embeddings = Vec::new();
    for info in source_face_infos.iter() {
        log_additional!(
            EVENT_ALIGN_FACE,
            "Aligning source face",
            filename = &info.source_filename
        );
        let source_img = &source_images[info.source_image_index];
        let source_aligned = alignment::align_face(source_img, &info.face, 112)?;
        let embedding = recognizer.extract_embedding(&source_aligned.aligned_image)?;
        source_embeddings.push(embedding);
    }

    // Swap each mapped face
    for mapping in &mappings {
        log_main!(
            "multi_face",
            "Swapping face",
            source_idx = mapping.source_idx,
            target_idx = mapping.target_idx
        );

        let source_info = &source_face_infos[mapping.source_idx];
        let source_img = &source_images[source_info.source_image_index];
        let source_embedding = &source_embeddings[mapping.source_idx];
        let target_face = &target_faces[mapping.target_idx];

        swap_single_pair(
            target_image,
            target_face,
            source_img,
            source_embedding,
            swapper,
            enhancer,
            landmark_detector,
            use_mouth_mask,
        )?;
    }

    Ok(())
}

/// Simple face swap between two images
///
/// # Arguments
/// * `source_path` - Path to source image (face to extract)
/// * `target_path` - Path to target image (face to replace)
/// * `output_path` - Path to save result
/// * `detector_model` - Path to detection model (det_10g.onnx)
/// * `recognizer_model` - Path to recognition model (w600k_r50.onnx)
/// * `swapper_model` - Path to swapper model (inswapper_128.onnx)
/// * `enhancer_model` - Optional path to enhancement model (GFPGAN of version
/// 1.4 is tested by me, but others may work too. If you are figured it out let me know)
/// * `landmark_model` - Optional path to 106-point landmark model (2d106det.onnx).
/// Required when `mouth_mask` is true.
/// * `mouth_mask` - Whether to apply mouth mask to preserve target's mouth expression
/// * `multi_face` - Whether to enable multi-face processing with interactive mapping
///
/// # Returns
/// `Ok(())` on success, error otherwise
pub fn swap_faces(
    source_path: &str,
    target_path: &str,
    output_path: &str,
    detector_model: &str,
    recognizer_model: &str,
    swapper_model: &str,
    enhancer_model: Option<&str>,
    landmark_model: Option<&str>,
    use_mouth_mask: bool,
    use_multi_face: bool,
) -> Result<()> {
    log_main!(
        "swap_init",
        "Initializing face swap",
        source = source_path,
        target = target_path,
        output = output_path,
        multi_face = use_multi_face
    );

    let mut detector = FaceDetector::new(detector_model)?;
    let mut recognizer = FaceRecognizer::new(recognizer_model)?;
    let mut swapper = FaceSwapper::new(swapper_model)?;
    let mut enhancer = enhancer_model
        .map(|path| FaceEnhancer::new(path))
        .transpose()?;
    let mut landmark_detector = if use_mouth_mask {
        let path = landmark_model.ok_or_else(|| {
            FaceSwapError::InvalidInput(
                "Landmark model path required when mouth mask is enabled".to_string(),
            )
        })?;
        Some(LandmarkDetector::new(path)?)
    } else {
        None
    };

    let source_paths: Vec<&str> = source_path.split(',').map(|s| s.trim()).collect();
    let mut source_images: Vec<Array3<u8>> = Vec::new();
    let mut all_source_faces: Vec<SourceFaceInfo> = Vec::new();

    for (img_idx, path) in source_paths.iter().enumerate() {
        log_additional!("load_image", "Loading source image", path = path);

        let img = img_io::load_image(path)?;
        let rgb = img_io::to_rgb8(&img);
        let array = rgb::rgb_to_array3(&rgb);

        let faces = detector.detect(&array, 0.5, 0.4)?;

        if faces.is_empty() {
            log_main!("warn", "No faces detected in source image", path = path);
        } else {
            log_main!(
                EVENT_FACE_DETECTED,
                "Detected faces in source image",
                path = path,
                count = faces.len()
            );

            let filename = std::path::Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path);

            for face in faces {
                all_source_faces.push(SourceFaceInfo {
                    face,
                    source_image_index: img_idx,
                    source_filename: filename.to_string(),
                });
            }
        }

        source_images.push(array);
    }

    if all_source_faces.is_empty() {
        return Err(FaceSwapError::NoFacesDetected);
    }

    log_main!(
        EVENT_FACE_DETECTED,
        "Total source faces across all images",
        count = all_source_faces.len()
    );

    let target_img = img_io::load_image(target_path)?;
    let target_rgb = img_io::to_rgb8(&target_img);
    let target_array = rgb::rgb_to_array3(&target_rgb);

    let target_faces = detector.detect(&target_array, 0.5, 0.4)?;
    if target_faces.is_empty() {
        return Err(FaceSwapError::NoFacesDetected);
    }
    log_main!(
        EVENT_FACE_DETECTED,
        "Detected target faces",
        count = target_faces.len()
    );

    let mut result = target_array.clone();

    // Branch: multi-face or single-face swap
    if use_multi_face && (all_source_faces.len() > 1 || target_faces.len() > 1) {
        log_main!(
            "multi_face",
            "Multi-face mode enabled",
            source_count = all_source_faces.len(),
            target_count = target_faces.len()
        );

        swap_multiple_faces(
            &all_source_faces,
            &target_faces,
            &source_images,
            &mut result,
            &mut recognizer,
            &mut swapper,
            &mut enhancer,
            &mut landmark_detector,
            use_mouth_mask,
        )?;
    } else {
        // Single face swap (use first face from each)
        let source_face_info = &all_source_faces[0];
        let source_face = &source_face_info.face;
        let source_image = &source_images[source_face_info.source_image_index];
        let target_face = &target_faces[0];

        if all_source_faces.len() > 1 {
            log_main!(
                EVENT_FACE_DETECTED,
                "Multiple source faces detected, using highest score",
                filename = &source_face_info.source_filename,
                score = source_face.det_score
            );
        }

        if target_faces.len() > 1 {
            log_main!(
                EVENT_FACE_DETECTED,
                "Multiple target faces, using highest score",
                score = target_face.det_score
            );
        }

        log_additional!(EVENT_ALIGN_FACE, "Aligning source face");
        let source_aligned = alignment::align_face(source_image, source_face, 112)?;
        let source_embedding = recognizer.extract_embedding(&source_aligned.aligned_image)?;

        swap_single_pair(
            &mut result,
            target_face,
            source_image,
            &source_embedding,
            &mut swapper,
            &mut enhancer,
            &mut landmark_detector,
            use_mouth_mask,
        )?;
    }

    let result_img = rgb::array3_to_rgb(&result);
    img_io::save_image(&result_img, output_path)?;

    log_main!(EVENT_COMPLETE, "Face swap completed successfully");
    Ok(())
}
