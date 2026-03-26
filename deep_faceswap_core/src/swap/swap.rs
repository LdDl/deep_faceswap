//! Main face swap operation

use crate::alignment;
use crate::detection::FaceDetector;
use crate::enhancer::FaceEnhancer;
use crate::landmark::LandmarkDetector;
use crate::mouth_mask;
use crate::multi_face;
use crate::recognition::FaceRecognizer;
use crate::swapper::FaceSwapper;
use crate::types::{ClusterMapping, DetectedFace, FaceSwapError, Result, SourceFaceInfo};
use crate::utils::image as img_io;
use crate::utils::rgb;
use crate::verbose::{EVENT_ALIGN_FACE, EVENT_COMPLETE, EVENT_FACE_DETECTED, EVENT_PASTE_BACK};
use crate::video::{
    build_cluster_info, build_face_lookup, cluster_faces, extract_frames, scan_frames_for_faces,
};
use crate::{log_additional, log_main};
use ndarray::Array3;
use std::collections::HashMap;
use std::fs::create_dir_all;
use std::time::Instant;

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
/// Modified target_image with swapped face and elapsed time in seconds
pub fn swap_single_pair(
    target_image: &mut Array3<u8>,
    target_face: &DetectedFace,
    _source_image: &Array3<u8>,
    source_embedding: &[f32],
    swapper: &mut FaceSwapper,
    enhancer: &mut Option<FaceEnhancer>,
    landmark_detector: &mut Option<LandmarkDetector>,
    use_mouth_mask: bool,
) -> Result<f64> {
    let start_time = Instant::now();

    // Detect 106 landmarks on target face before swap (needed for mouth mask)
    let mouth_mask_data = if use_mouth_mask {
        if let Some(ref mut lm_detector) = landmark_detector {
            log_additional!("mouth_mask", "Detecting 106 landmarks on target face");
            let landmarks = lm_detector.detect(target_image, target_face)?;
            let data = mouth_mask::create_mouth_mask(target_image, &landmarks)?;
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
    alignment::paste_back_inplace(target_image, &swapped_face, &target_aligned.transform, 128)?;

    // Apply mouth mask after swap but before enhancement
    if let Some(ref data) = mouth_mask_data {
        log_additional!("mouth_mask", "Applying mouth mask");
        mouth_mask::apply_mouth_mask(target_image, data);
    }

    // Enhance face if enhancer is provided
    if let Some(ref mut enh) = enhancer {
        log_additional!("enhance_face", "Enhancing face at original resolution");

        // Align target face from current image to 512x512 for enhancement
        let target_aligned_512 = alignment::align_face(target_image, target_face, 512)?;

        // Enhance the 512x512 aligned face
        let enhanced_512 = enh.enhance(&target_aligned_512.aligned_image)?;

        // Paste enhanced face back in-place
        alignment::paste_back_inplace(
            target_image,
            &enhanced_512,
            &target_aligned_512.transform,
            512,
        )?;
    }

    Ok(start_time.elapsed().as_secs_f64())
}

/// Swap multiple faces using explicit mappings (no interactive prompts)
///
/// This is the programmatic API suitable external usage (e.g. for REST endpoints).
/// Extracts embeddings for all source faces, then applies each mapping.
///
/// # Arguments
/// * `source_face_infos` - Source faces with origin metadata
/// * `target_faces` - All detected faces in target image
/// * `source_images` - All loaded source images
/// * `target_image` - Mutable target image (HWC, RGB, u8)
/// * `mappings` - Explicit source->target face mappings
/// * `recognizer` - Face recognizer for embedding extraction
/// * `swapper` - Face swapper model
/// * `enhancer` - Optional face enhancer model
/// * `landmark_detector` - Optional 106-point landmark detector
/// * `use_mouth_mask` - Whether to apply mouth mask
///
/// # Returns
/// Elapsed time in seconds
pub fn swap_with_mappings(
    source_face_infos: &[SourceFaceInfo],
    target_faces: &[DetectedFace],
    source_images: &[Array3<u8>],
    target_image: &mut Array3<u8>,
    mappings: &[crate::types::FaceMapping],
    recognizer: &mut FaceRecognizer,
    swapper: &mut FaceSwapper,
    enhancer: &mut Option<FaceEnhancer>,
    landmark_detector: &mut Option<LandmarkDetector>,
    use_mouth_mask: bool,
) -> Result<f64> {
    log_main!(
        "multi_face",
        "Processing face mappings",
        count = mappings.len()
    );

    let start_time = Instant::now();

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
    for mapping in mappings {
        log_additional!(
            "multi_face",
            "Swapping face",
            source_idx = mapping.source_idx,
            target_idx = mapping.target_idx
        );

        let source_info = &source_face_infos[mapping.source_idx];
        let source_img = &source_images[source_info.source_image_index];
        let source_embedding = &source_embeddings[mapping.source_idx];
        let target_face = &target_faces[mapping.target_idx];

        let _ = swap_single_pair(
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

    Ok(start_time.elapsed().as_secs_f64())
}

/// Swap multiple faces in target image based on interactive face mapping
///
/// CLI-oriented wrapper that builds face mappings via interactive prompts,
/// then delegates to swap_with_mappings.
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
) -> Result<f64> {
    // Build face mappings via interactive prompts
    let mappings = multi_face::build_face_mappings(
        source_face_infos,
        target_faces,
        source_images,
        target_image,
    )?;

    swap_with_mappings(
        source_face_infos,
        target_faces,
        source_images,
        target_image,
        &mappings,
        recognizer,
        swapper,
        enhancer,
        landmark_detector,
        use_mouth_mask,
    )
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

    let action_start = Instant::now();
    let source_paths: Vec<&str> = source_path.split(',').map(|s| s.trim()).collect();
    let mut source_images: Vec<Array3<u8>> = Vec::new();
    let mut all_source_faces: Vec<SourceFaceInfo> = Vec::new();

    for (img_idx, path) in source_paths.iter().enumerate() {
        log_additional!("load_image", "Loading source image", path = path);

        let img = img_io::load_image(path)?;
        let rgb = img_io::to_rgb8(&img);
        let array = rgb::rgb_to_array3(&rgb);

        let detect_start = Instant::now();
        let faces = detector.detect(&array, 0.5, 0.4)?;

        if faces.is_empty() {
            log_main!("warn", "No faces detected in source image", path = path);
        } else {
            log_main!(
                EVENT_FACE_DETECTED,
                "Detected faces in source image",
                path = path,
                count = faces.len(),
                elapsed_s = detect_start.elapsed().as_secs_f64()
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
        "load_source",
        "Loading source images complete",
        count = all_source_faces.len(),
        elapsed_s = action_start.elapsed().as_secs_f64()
    );

    let action_start = Instant::now();
    let target_img = img_io::load_image(target_path)?;
    let target_rgb = img_io::to_rgb8(&target_img);
    let target_array = rgb::rgb_to_array3(&target_rgb);

    let target_faces = detector.detect(&target_array, 0.5, 0.4)?;
    if target_faces.is_empty() {
        return Err(FaceSwapError::NoFacesDetected);
    }
    log_main!(
        "load_target",
        "Loading target image complete",
        count = target_faces.len(),
        elapsed_s = action_start.elapsed().as_secs_f64()
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

        let elapsed_s = swap_multiple_faces(
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
        log_main!(
            "multi_face",
            "Multi-face swap complete",
            elapsed_s = elapsed_s
        );
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

        let elapsed_s = swap_single_pair(
            &mut result,
            target_face,
            source_image,
            &source_embedding,
            &mut swapper,
            &mut enhancer,
            &mut landmark_detector,
            use_mouth_mask,
        )?;
        log_main!("face_swap", "Face swap complete", elapsed_s = elapsed_s);
    }

    let result_img = rgb::array3_to_rgb(&result);
    img_io::save_image(&result_img, output_path)?;

    log_main!(EVENT_COMPLETE, "Face swap completed successfully");
    Ok(())
}

/// Swap faces in video files
///
/// # Arguments
/// * `source_path` - Path to source image(s) - single path or comma-separated list
/// * `target_path` - Path to target video
/// * `output_path` - Path to output video
/// * `detector_path` - Path to detection model
/// * `recognizer_path` - Path to recognition model
/// * `swapper_path` - Path to swapper model
/// * `enhancer_path` - Optional path to enhancement model
/// * `landmark_path` - Optional path to 106-point landmark model
/// * `use_mouth_mask` - Enable mouth mask
/// * `use_multi_face` - Enable multi-face processing with interactive mapping
///
/// # Returns
/// Ok(()) on success
pub fn swap_video(
    source_path: &str,
    target_path: &str,
    output_path: &str,
    detector_path: &str,
    recognizer_path: &str,
    swapper_path: &str,
    enhancer_path: Option<&str>,
    landmark_path: Option<&str>,
    use_mouth_mask: bool,
    use_multi_face: bool,
    video_tmp_dir: Option<&str>,
) -> Result<()> {
    log_main!("video_processing", "Starting video face swap");

    log_additional!("load_models", "Loading models");
    let mut detector = FaceDetector::new(detector_path)?;
    let mut recognizer = FaceRecognizer::new(recognizer_path)?;
    let mut swapper = FaceSwapper::new(swapper_path)?;

    let mut enhancer = if let Some(path) = enhancer_path {
        log_additional!("load_models", "Loading enhancement model");
        Some(FaceEnhancer::new(path)?)
    } else {
        None
    };

    let mut landmark_detector = if let Some(path) = landmark_path {
        log_additional!("load_models", "Loading landmark model");
        Some(LandmarkDetector::new(path)?)
    } else {
        None
    };

    // Load and process source images
    let action_start = Instant::now();
    let source_paths: Vec<&str> = source_path.split(',').map(|s| s.trim()).collect();
    let mut source_images: Vec<Array3<u8>> = Vec::new();
    let mut all_source_faces: Vec<SourceFaceInfo> = Vec::new();

    for (img_idx, path) in source_paths.iter().enumerate() {
        log_additional!("load_image", "Loading source image", path = path);

        let img = img_io::load_image(path)?;
        let rgb = img_io::to_rgb8(&img);
        let array = rgb::rgb_to_array3(&rgb);

        let detect_start = Instant::now();
        let faces = detector.detect(&array, 0.5, 0.4)?;

        if faces.is_empty() {
            log_main!("warn", "No faces detected in source image", path = path);
        } else {
            log_main!(
                EVENT_FACE_DETECTED,
                "Detected faces in source image",
                path = path,
                count = faces.len(),
                elapsed_s = detect_start.elapsed().as_secs_f64()
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
        "load_source",
        "Loading source images complete",
        count = all_source_faces.len(),
        elapsed_s = action_start.elapsed().as_secs_f64()
    );

    // Extract embeddings from source faces (cached for all frames)
    let action_start = Instant::now();
    let mut source_embeddings = Vec::new();

    if use_multi_face {
        // Multi-face mode: extract embeddings for all source faces
        for info in all_source_faces.iter() {
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
        log_main!(
            "extract_embeddings",
            "Extracting source face embeddings complete",
            count = source_embeddings.len(),
            elapsed_s = action_start.elapsed().as_secs_f64()
        );
    } else {
        // Single-face mode: extract embedding for first source face
        let source_face_info = &all_source_faces[0];
        let source_face = &source_face_info.face;
        let source_image = &source_images[source_face_info.source_image_index];

        if all_source_faces.len() > 1 {
            log_main!(
                EVENT_FACE_DETECTED,
                "Multiple source faces detected, using highest score",
                filename = &source_face_info.source_filename,
                score = source_face.det_score
            );
        }

        let source_aligned = alignment::align_face(source_image, source_face, 112)?;
        let source_embedding = recognizer.extract_embedding(&source_aligned.aligned_image)?;
        source_embeddings.push(source_embedding);
        log_main!(
            "extract_embeddings",
            "Extracting source face embeddings complete",
            elapsed_s = action_start.elapsed().as_secs_f64()
        );
    }

    // Extract frames from video
    let action_start = Instant::now();
    let base_tmp_dir = video_tmp_dir.unwrap_or("./tmp");
    let frames_dir = format!("{}/video_frames", base_tmp_dir);
    let processed_frames_dir = format!("{}/video_frames_processed", base_tmp_dir);

    create_dir_all(&frames_dir)?;
    create_dir_all(&processed_frames_dir)?;

    let frame_paths = extract_frames(target_path, &frames_dir)?;
    let total_frames = frame_paths.len();

    log_main!(
        "video_extraction",
        "Extraction complete",
        total_frames = total_frames,
        elapsed_s = action_start.elapsed().as_secs_f64()
    );

    log_main!("video_processing", "Starting frame processing");

    let log_interval = std::cmp::max(10, total_frames / 10);
    let processing_start = Instant::now();

    if use_multi_face {
        // Multi-face mode with clustering
        log_main!(
            "multi_face",
            "Multi-face mode for video",
            source_count = all_source_faces.len()
        );

        // scan all frames and extract face embeddings
        let scan_start = Instant::now();
        let mut face_records = scan_frames_for_faces(&frame_paths, &mut detector, &mut recognizer)?;
        log_main!(
            "video_analysis",
            "Frame scanning complete",
            total_faces = face_records.len(),
            elapsed_s = scan_start.elapsed().as_secs_f64()
        );

        // cluster faces across video
        let cluster_start = Instant::now();
        let max_k = 10;
        let centroids = cluster_faces(&mut face_records, max_k)?;
        log_main!(
            "video_analysis",
            "Face clustering complete",
            clusters = centroids.nrows(),
            elapsed_s = cluster_start.elapsed().as_secs_f64()
        );

        // Build cluster info and face lookup
        let cluster_infos = build_cluster_info(&face_records, &centroids)?;
        let face_lookup = build_face_lookup(&face_records);

        // interactive mapping (source faces -> clusters)
        let cluster_mappings = multi_face::build_cluster_mappings(
            &all_source_faces,
            &cluster_infos,
            &source_images,
            &frame_paths,
        )?;
        log_main!(
            "multi_face",
            "Cluster mapping created",
            mappings = cluster_mappings.len()
        );

        // process all frames with cluster mappings
        for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
            let frame_img = img_io::load_image(frame_path)?;
            let frame_rgb = img_io::to_rgb8(&frame_img);
            let mut frame_array = rgb::rgb_to_array3(&frame_rgb);

            // apply each mapping
            for mapping in &cluster_mappings {
                let key = (frame_idx, mapping.cluster_id);
                if let Some(target_faces) = face_lookup.get(&key) {
                    // swap all faces in this cluster at specific frame
                    for target_face in target_faces {
                        let source_info = &all_source_faces[mapping.source_idx];
                        let source_img = &source_images[source_info.source_image_index];
                        let source_embedding = &source_embeddings[mapping.source_idx];

                        let _ = swap_single_pair(
                            &mut frame_array,
                            target_face,
                            source_img,
                            source_embedding,
                            &mut swapper,
                            &mut enhancer,
                            &mut landmark_detector,
                            use_mouth_mask,
                        )?;
                    }
                }
            }

            let processed_frame = rgb::array3_to_rgb(&frame_array);
            let output_frame_path = format!("{}/frame_{:06}.png", processed_frames_dir, frame_idx);
            img_io::save_image_quiet(&processed_frame, &output_frame_path)?;

            if (frame_idx + 1) % log_interval == 0 {
                log_main!(
                    "video_processing",
                    "Processing frames",
                    processed = frame_idx + 1,
                    total = total_frames,
                    elapsed_s = processing_start.elapsed().as_secs_f64()
                );
            }
        }
    } else {
        // Single-face mode with embedding-based tracking
        let source_face_info = &all_source_faces[0];
        let source_image = &source_images[source_face_info.source_image_index];
        let source_embedding = &source_embeddings[0];

        log_main!(
            "video_processing",
            "Single-face mode with embedding-based matching"
        );

        // Extract reference embedding from first detected face
        let mut reference_embedding: Option<Vec<f32>> = None;

        for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
            let frame_img = img_io::load_image(frame_path)?;
            let frame_rgb = img_io::to_rgb8(&frame_img);
            let mut frame_array = rgb::rgb_to_array3(&frame_rgb);

            let target_faces = detector.detect(&frame_array, 0.5, 0.4)?;

            if !target_faces.is_empty() {
                // Initialize reference embedding from first frame
                if reference_embedding.is_none() && !target_faces.is_empty() {
                    let first_face = &target_faces[0];
                    let aligned = alignment::align_face(&frame_array, first_face, 112)?;
                    reference_embedding =
                        Some(recognizer.extract_embedding(&aligned.aligned_image)?);
                }

                // find face most similar to reference embedding
                if let Some(ref ref_emb) = reference_embedding {
                    if let Some(target_face) = crate::utils::embedding::find_most_similar_face(
                        &target_faces,
                        ref_emb,
                        &frame_array,
                        &mut recognizer,
                        0.3,
                    )? {
                        let _ = swap_single_pair(
                            &mut frame_array,
                            target_face,
                            source_image,
                            source_embedding,
                            &mut swapper,
                            &mut enhancer,
                            &mut landmark_detector,
                            use_mouth_mask,
                        )?;
                    }
                }
            }

            let processed_frame = rgb::array3_to_rgb(&frame_array);
            let output_frame_path = format!("{}/frame_{:06}.png", processed_frames_dir, frame_idx);
            img_io::save_image_quiet(&processed_frame, &output_frame_path)?;

            if (frame_idx + 1) % log_interval == 0 {
                log_main!(
                    "video_processing",
                    "Processing frames",
                    processed = frame_idx + 1,
                    total = total_frames,
                    elapsed_s = processing_start.elapsed().as_secs_f64()
                );
            }
        }
    }

    log_main!(
        "video_processing",
        "All frames processed",
        total = total_frames,
        elapsed_s = processing_start.elapsed().as_secs_f64()
    );

    // Encode processed frames back to video
    let action_start = Instant::now();
    crate::video::encode_video(&processed_frames_dir, output_path, target_path)?;
    log_main!(
        "video_encoding",
        "Encoding complete",
        elapsed_s = action_start.elapsed().as_secs_f64()
    );

    // Cleanup temporary directories
    log_additional!("cleanup", "Cleaning up temporary files");
    std::fs::remove_dir_all(&frames_dir)?;
    std::fs::remove_dir_all(&processed_frames_dir)?;

    log_main!(EVENT_COMPLETE, "Video face swap completed successfully");
    Ok(())
}

/// Process video frames with cluster mappings (no interactive prompts)
///
/// Programmatic API for video frame processing. Applies cluster mappings
/// to each frame, calling swap_single_pair for each matched face.
/// Optionally reports progress via callback.
///
/// # Arguments
/// * `frame_paths` - Paths to extracted video frames
/// * `processed_frames_dir` - Directory to save processed frames
/// * `cluster_mappings` - Source-to-cluster mappings
/// * `face_lookup` - Precomputed (frame_idx, cluster_id) -> faces lookup
/// * `all_source_faces` - Source face metadata
/// * `source_images` - Loaded source images
/// * `source_embeddings` - Pre-extracted source embeddings
/// * `swapper` - Face swapper model
/// * `enhancer` - Optional face enhancer model
/// * `landmark_detector` - Optional 106-point landmark detector
/// * `use_mouth_mask` - Whether to apply mouth mask
/// * `progress_callback` - Optional (current_frame, total_frames) callback
pub fn swap_video_frames_with_mappings(
    frame_paths: &[String],
    processed_frames_dir: &str,
    cluster_mappings: &[ClusterMapping],
    face_lookup: &HashMap<(usize, usize), Vec<DetectedFace>>,
    all_source_faces: &[SourceFaceInfo],
    source_images: &[Array3<u8>],
    source_embeddings: &[Vec<f32>],
    swapper: &mut FaceSwapper,
    enhancer: &mut Option<FaceEnhancer>,
    landmark_detector: &mut Option<LandmarkDetector>,
    use_mouth_mask: bool,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> Result<()> {
    let total_frames = frame_paths.len();
    let log_interval = std::cmp::max(10, total_frames / 10);
    let processing_start = Instant::now();

    for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
        let frame_img = img_io::load_image(frame_path)?;
        let frame_rgb = img_io::to_rgb8(&frame_img);
        let mut frame_array = rgb::rgb_to_array3(&frame_rgb);

        for mapping in cluster_mappings {
            let key = (frame_idx, mapping.cluster_id);
            if let Some(target_faces) = face_lookup.get(&key) {
                for target_face in target_faces {
                    let source_info = &all_source_faces[mapping.source_idx];
                    let source_img = &source_images[source_info.source_image_index];
                    let source_embedding = &source_embeddings[mapping.source_idx];

                    let _ = swap_single_pair(
                        &mut frame_array,
                        target_face,
                        source_img,
                        source_embedding,
                        swapper,
                        enhancer,
                        landmark_detector,
                        use_mouth_mask,
                    )?;
                }
            }
        }

        let processed_frame = rgb::array3_to_rgb(&frame_array);
        let output_frame_path = format!("{}/frame_{:06}.png", processed_frames_dir, frame_idx);
        img_io::save_image_quiet(&processed_frame, &output_frame_path)?;

        if let Some(ref cb) = progress_callback {
            cb(frame_idx + 1, total_frames);
        }

        if (frame_idx + 1) % log_interval == 0 {
            log_main!(
                "video_processing",
                "Processing frames",
                processed = frame_idx + 1,
                total = total_frames,
                elapsed_s = processing_start.elapsed().as_secs_f64()
            );
        }
    }

    log_main!(
        "video_processing",
        "All frames processed",
        total = total_frames,
        elapsed_s = processing_start.elapsed().as_secs_f64()
    );

    Ok(())
}

/// Process video frames in single-face mode (no interactive prompts)
///
/// Programmatic API for single-face video processing. Uses embedding-based
/// face tracking to find the target face in each frame.
///
/// # Arguments
/// * `frame_paths` - Paths to extracted video frames
/// * `processed_frames_dir` - Directory to save processed frames
/// * `source_image` - Source image (HWC, RGB, u8)
/// * `source_embedding` - Pre-extracted source face embedding
/// * `detector` - Face detector model
/// * `recognizer` - Face recognizer for embedding matching
/// * `swapper` - Face swapper model
/// * `enhancer` - Optional face enhancer model
/// * `landmark_detector` - Optional 106-point landmark detector
/// * `use_mouth_mask` - Whether to apply mouth mask
/// * `progress_callback` - Optional (current_frame, total_frames) callback
pub fn swap_video_frames_single(
    frame_paths: &[String],
    processed_frames_dir: &str,
    source_image: &Array3<u8>,
    source_embedding: &[f32],
    detector: &mut FaceDetector,
    recognizer: &mut FaceRecognizer,
    swapper: &mut FaceSwapper,
    enhancer: &mut Option<FaceEnhancer>,
    landmark_detector: &mut Option<LandmarkDetector>,
    use_mouth_mask: bool,
    progress_callback: Option<&dyn Fn(usize, usize)>,
) -> Result<()> {
    let total_frames = frame_paths.len();
    let log_interval = std::cmp::max(10, total_frames / 10);
    let processing_start = Instant::now();

    let mut reference_embedding: Option<Vec<f32>> = None;

    for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
        let frame_img = img_io::load_image(frame_path)?;
        let frame_rgb = img_io::to_rgb8(&frame_img);
        let mut frame_array = rgb::rgb_to_array3(&frame_rgb);

        let target_faces = detector.detect(&frame_array, 0.5, 0.4)?;

        if !target_faces.is_empty() {
            // Initialize reference embedding from first frame with a face
            if reference_embedding.is_none() {
                let first_face = &target_faces[0];
                let aligned = alignment::align_face(&frame_array, first_face, 112)?;
                reference_embedding = Some(recognizer.extract_embedding(&aligned.aligned_image)?);
            }

            if let Some(ref ref_emb) = reference_embedding {
                if let Some(target_face) = crate::utils::embedding::find_most_similar_face(
                    &target_faces,
                    ref_emb,
                    &frame_array,
                    recognizer,
                    0.3,
                )? {
                    let _ = swap_single_pair(
                        &mut frame_array,
                        target_face,
                        source_image,
                        source_embedding,
                        swapper,
                        enhancer,
                        landmark_detector,
                        use_mouth_mask,
                    )?;
                }
            }
        }

        let processed_frame = rgb::array3_to_rgb(&frame_array);
        let output_frame_path = format!("{}/frame_{:06}.png", processed_frames_dir, frame_idx);
        img_io::save_image_quiet(&processed_frame, &output_frame_path)?;

        if let Some(ref cb) = progress_callback {
            cb(frame_idx + 1, total_frames);
        }

        if (frame_idx + 1) % log_interval == 0 {
            log_main!(
                "video_processing",
                "Processing frames",
                processed = frame_idx + 1,
                total = total_frames,
                elapsed_s = processing_start.elapsed().as_secs_f64()
            );
        }
    }

    log_main!(
        "video_processing",
        "All frames processed",
        total = total_frames,
        elapsed_s = processing_start.elapsed().as_secs_f64()
    );

    Ok(())
}
