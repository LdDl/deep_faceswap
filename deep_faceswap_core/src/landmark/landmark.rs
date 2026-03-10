//! 106-point facial landmark detection
//!
//! This module detects 106 facial landmark points using the 2d106det model
//! from the InsightFace buffalo_l package.
//!
//! # Model details
//! - Model: 2d106det.onnx
//! - Input size: 192x192 RGB images
//! - Input normalization: raw pixels (0-255), model has internal Sub/Mul nodes
//! - Output: 212 values (106 points x 2 coordinates)
//!
//! # Usage
//! The 106-point landmarks are used for mouth mask feature.
//! Standard 5-point landmarks from the face detector are not sufficient
//! for precise mouth region identification.

use crate::types::{DetectedFace, FaceSwapError, Result};
use crate::utils::transform::warp_affine;
use crate::verbose::{get_verbose_level, VerboseLevel, EVENT_LOAD_MODEL};
use crate::{log_additional, log_main};
use ndarray::{Array2, Array4};
use ort::{
    inputs,
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::path::Path;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

const LANDMARK_INPUT_SIZE: usize = 192;
const LANDMARK_NUM_POINTS: usize = 106;

pub const EVENT_DETECT_LANDMARKS: &str = "detect_landmarks";

pub struct LandmarkDetector {
    session: Session,
}

impl LandmarkDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        if !Path::new(model_path).exists() {
            return Err(FaceSwapError::ModelNotFound(model_path.to_string()));
        }

        log_main!(
            EVENT_LOAD_MODEL,
            "Loading landmark detector",
            path = model_path
        );

        let ort_log_level = match get_verbose_level() {
            VerboseLevel::None => LogLevel::Fatal,
            VerboseLevel::Main => LogLevel::Fatal,
            VerboseLevel::Additional => LogLevel::Warning,
            VerboseLevel::All => LogLevel::Verbose,
        };

        #[cfg(feature = "cuda")]
        let session = Session::builder()
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_log_level(ort_log_level)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

        #[cfg(not(feature = "cuda"))]
        let session = Session::builder()
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_log_level(ort_log_level)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

        Ok(Self { session })
    }

    /// Detect 106 facial landmarks for a face in the image.
    ///
    /// The detection pipeline:
    /// 1. Crop face region from image using bbox with 1.5x padding
    /// 2. Build affine transform to 192x192
    /// 3. Build NCHW tensor with RGB channel order, raw pixel values
    /// 4. Run inference -> 212 floats
    /// 5. Reshape to 106x2, denormalize: (val + 1) * 96
    /// 6. Apply inverse transform to get original image coordinates
    pub fn detect(
        &mut self,
        image: &ndarray::Array3<u8>,
        face: &DetectedFace,
    ) -> Result<Vec<[f32; 2]>> {
        log_additional!(EVENT_DETECT_LANDMARKS, "Detecting 106 landmarks");

        let (img_h, img_w, _) = image.dim();
        let bbox = &face.bbox;

        // Calculate face center and scale
        let w = bbox.x2 - bbox.x1;
        let h = bbox.y2 - bbox.y1;
        let cx = (bbox.x2 + bbox.x1) / 2.0;
        let cy = (bbox.y2 + bbox.y1) / 2.0;
        let max_side = w.max(h);
        let scale = LANDMARK_INPUT_SIZE as f32 / (max_side * 1.5);

        // Build affine transform: original image -> 192x192 crop
        // new_x = px * scale - cx * scale + 96
        // new_y = py * scale - cy * scale + 96
        let half_out = LANDMARK_INPUT_SIZE as f32 / 2.0;
        let tx = half_out - cx * scale;
        let ty = half_out - cy * scale;

        let transform = Array2::from_shape_vec((2, 3), vec![scale, 0.0, tx, 0.0, scale, ty])
            .map_err(|e| FaceSwapError::ProcessingError(format!("{}", e)))?;

        // Warp image to 192x192
        // warp_affine expects inverse transform (aligned -> original mapping)
        let inv_crop_transform = crate::utils::transform::invert_affine_transform(&transform)?;
        let cropped = warp_affine(image, &inv_crop_transform, LANDMARK_INPUT_SIZE)?;

        // Build NCHW tensor with RGB order, raw pixel values (0-255)
        // The 2d106det ONNX model has internal Sub/Mul normalization nodes,
        // so it expects raw pixel values, not pre-normalized ones.
        let mut input_tensor =
            Array4::<f32>::zeros((1, 3, LANDMARK_INPUT_SIZE, LANDMARK_INPUT_SIZE));
        for y in 0..LANDMARK_INPUT_SIZE {
            for x in 0..LANDMARK_INPUT_SIZE {
                let r = cropped[[y, x, 0]] as f32;
                let g = cropped[[y, x, 1]] as f32;
                let b = cropped[[y, x, 2]] as f32;
                // RGB order! (not as OpenCV's one which has for dnn.blobFromImage:
                // swapRB=True to convert BGR to RGB)
                input_tensor[[0, 0, y, x]] = r;
                input_tensor[[0, 1, y, x]] = g;
                input_tensor[[0, 2, y, x]] = b;
            }
        }

        // Run inference
        let pred = {
            let outputs = self
                .session
                .run(inputs!["data" => TensorRef::from_array_view(input_tensor.view())?])
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

            outputs["fc1"]
                .try_extract_array::<f32>()
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                .into_owned()
        };

        // Reshape to 106x2 and denormalize
        // Model output is in [-1, 1] range, denormalize: (val + 1) * 96
        let half_size = (LANDMARK_INPUT_SIZE / 2) as f32;
        let mut landmarks_192 = vec![[0.0f32; 2]; LANDMARK_NUM_POINTS];
        for i in 0..LANDMARK_NUM_POINTS {
            landmarks_192[i][0] = (pred[[0, i * 2]] + 1.0) * half_size;
            landmarks_192[i][1] = (pred[[0, i * 2 + 1]] + 1.0) * half_size;
        }

        // Invert affine transform to map landmarks back to original image space
        let inv_transform = crate::utils::transform::invert_affine_transform(&transform)?;
        let mut landmarks = vec![[0.0f32; 2]; LANDMARK_NUM_POINTS];
        for i in 0..LANDMARK_NUM_POINTS {
            let px = landmarks_192[i][0];
            let py = landmarks_192[i][1];
            let orig_x =
                inv_transform[[0, 0]] * px + inv_transform[[0, 1]] * py + inv_transform[[0, 2]];
            let orig_y =
                inv_transform[[1, 0]] * px + inv_transform[[1, 1]] * py + inv_transform[[1, 2]];
            landmarks[i][0] = orig_x.clamp(0.0, (img_w - 1) as f32);
            landmarks[i][1] = orig_y.clamp(0.0, (img_h - 1) as f32);
        }

        Ok(landmarks)
    }
}
