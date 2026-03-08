//! Main face swap operation

use crate::alignment;
use crate::detection::FaceDetector;
use crate::enhancer::FaceEnhancer;
use crate::landmark::LandmarkDetector;
use crate::mouth_mask;
use crate::recognition::FaceRecognizer;
use crate::swapper::FaceSwapper;
use crate::types::{FaceSwapError, Result};
use crate::utils::image as img_io;
use crate::utils::rgb;
use crate::verbose::{EVENT_ALIGN_FACE, EVENT_COMPLETE, EVENT_FACE_DETECTED, EVENT_PASTE_BACK};
use crate::{log_additional, log_main};

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
) -> Result<()> {
    log_main!(
        "swap_init",
        "Initializing face swap",
        source = source_path,
        target = target_path,
        output = output_path
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

    let source_img = img_io::load_image(source_path)?;
    let source_rgb = img_io::to_rgb8(&source_img);
    let source_array = rgb::rgb_to_array3(&source_rgb);

    let target_img = img_io::load_image(target_path)?;
    let target_rgb = img_io::to_rgb8(&target_img);
    let target_array = rgb::rgb_to_array3(&target_rgb);

    let source_faces = detector.detect(&source_array, 0.5, 0.4)?;
    if source_faces.is_empty() {
        return Err(FaceSwapError::NoFacesDetected);
    }
    // Take the first face (highest confidence score)
    let source_face = &source_faces[0];
    if source_faces.len() > 1 {
        log_main!(
            EVENT_FACE_DETECTED,
            "Multiple faces detected in source, using face with highest score",
            count = source_faces.len(),
            score = source_face.det_score
        );
    } else {
        log_additional!(
            EVENT_FACE_DETECTED,
            "Found source face",
            score = source_face.det_score
        );
    }

    let target_faces = detector.detect(&target_array, 0.5, 0.4)?;
    if target_faces.is_empty() {
        return Err(FaceSwapError::NoFacesDetected);
    }
    // Take the first face (highest confidence score)
    let target_face = &target_faces[0];
    if target_faces.len() > 1 {
        log_main!(
            EVENT_FACE_DETECTED,
            "Multiple faces detected in target, using face with highest score",
            count = target_faces.len(),
            score = target_face.det_score
        );
    } else {
        log_additional!(
            EVENT_FACE_DETECTED,
            "Found target face",
            score = target_face.det_score
        );
    }

    // Detect 106 landmarks on target face before swap (needed for mouth mask)
    let mouth_mask_data = if let Some(ref mut lm_detector) = landmark_detector {
        log_additional!("mouth_mask", "Detecting 106 landmarks on target face");
        let landmarks = lm_detector.detect(&target_array, target_face)?;
        let (frame_h, frame_w, _) = target_array.dim();
        let data = mouth_mask::create_mouth_mask(&target_array, &landmarks, frame_h, frame_w)?;
        Some(data)
    } else {
        None
    };

    log_additional!(EVENT_ALIGN_FACE, "Aligning source face");
    let source_aligned = alignment::align_face(&source_array, source_face, 112)?;
    let source_embedding = recognizer.extract_embedding(&source_aligned.aligned_image)?;

    log_additional!(EVENT_ALIGN_FACE, "Aligning target face");
    let target_aligned = alignment::align_face(&target_array, target_face, 128)?;

    let swapped_face = swapper.swap(&target_aligned.aligned_image, &source_embedding)?;

    log_additional!(EVENT_PASTE_BACK, "Pasting swapped face back");
    let mut result = alignment::paste_back(
        &target_array,
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
    // Extract face region at original resolution and enhance
    if let Some(ref mut enhancer) = enhancer {
        log_additional!("enhance_face", "Enhancing face at original resolution");

        // Align target face from result image to 512x512 for enhancement
        let target_aligned_512 = alignment::align_face(&result, target_face, 512)?;

        // @debug: Convert aligned face to RGB image for debug
        // let aligned_512_rgb = rgb::array_to_rgb(&target_aligned_512.aligned_image)?;
        // img_io::save_image(&aligned_512_rgb, "/tmp/debug_before_gfpgan.jpg")?;

        // Enhance the 512x512 aligned face
        let enhanced_512 = enhancer.enhance(&target_aligned_512.aligned_image)?;

        // @debug: Convert back to RGB for debug
        // let enhanced_img_512 = rgb::array_to_rgb(&enhanced_512)?;
        // img_io::save_image(&enhanced_img_512, "/tmp/debug_after_gfpgan.jpg")?;

        // Paste enhanced face back into result
        result = alignment::paste_back(
            &result,
            &enhanced_512,
            &target_aligned_512.transform,
            &target_aligned_512.face.bbox,
            512,
        )?;
    }

    let result_img = rgb::array3_to_rgb(&result);
    img_io::save_image(&result_img, output_path)?;

    log_main!(EVENT_COMPLETE, "Face swap completed successfully");
    Ok(())
}
