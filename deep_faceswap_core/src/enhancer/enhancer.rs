//! Face enhancement using GFPGAN
//!
//! This module implements face enhancement using the GFPGAN model, which improves
//! the quality of face images by restoring details and fixing artifacts.
//!
//! # Model details
//! - Model: GFPGANv1.4.onnx
//! - Input size: 512x512 RGB images
//! - Input normalization: [-1, 1]
//! - Output: Enhanced 512x512 RGB images
//! - Output normalization: [-1, 1]
//!
//! # Usage
//! The enhancer is optional in the face swap pipeline. It is applied after the face
//! swap but before pasting back to the original image.

use crate::types::{FaceSwapError, Result};
use crate::verbose::{get_verbose_level, VerboseLevel, EVENT_ENHANCE_FACE, EVENT_LOAD_MODEL};
use crate::{log_additional, log_main};
use ndarray::Array4;
use ort::{
    inputs,
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::path::Path;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

const ENHANCE_SIZE: usize = 512;

/// Face enhancer model using GFPGAN architecture
///
/// This enhancer restores face details and improves image quality using a generative
/// model trained specifically for face enhancement.
///
/// # Implementation details
/// - Uses ONNX Runtime for model inference
/// - Requires 512x512 face images
/// - Input/output normalized to [-1, 1] range
/// - Works on RGB channel order
///
/// # Example
/// ```ignore
/// use deep_faceswap_core::enhancer::FaceEnhancer;
///
/// let mut enhancer = FaceEnhancer::new("models/GFPGANv1.4.onnx")?;
/// let enhanced = enhancer.enhance(&face_image)?;
/// ```
pub struct FaceEnhancer {
    /// ONNX Runtime session for the face enhancement model
    session: Session,
}

impl FaceEnhancer {
    /// Load face enhancer from ONNX model
    ///
    /// Loads the GFPGAN model for face enhancement.
    ///
    /// # Arguments
    /// * `model_path` - Path to GFPGANv1.4.onnx model file
    ///
    /// # Returns
    /// Initialized face enhancer ready for inference
    ///
    /// # Errors
    /// - `ModelNotFound` if model file doesn't exist
    /// - `Ort` if ONNX Runtime initialization fails
    pub fn new(model_path: &str) -> Result<Self> {
        if !Path::new(model_path).exists() {
            return Err(FaceSwapError::ModelNotFound(model_path.to_string()));
        }

        log_main!(EVENT_LOAD_MODEL, "Loading face enhancer", path = model_path);

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

    /// Enhance face image quality
    ///
    /// Takes a face image and enhances it using GFPGAN. The input should be a
    /// 512x512 face image normalized to [0, 1] range.
    ///
    /// # Arguments
    /// * `face_image` - Face image as NCHW array (1, 3, 512, 512), normalized to [0, 1]
    ///
    /// # Returns
    /// Enhanced face as NCHW array (1, 3, 512, 512), normalized to [0, 1]
    ///
    /// # Errors
    /// - `InvalidInput` if face image shape is not (1, 3, 512, 512)
    /// - `Ort` if model inference fails
    /// - `ProcessingError` if output shape conversion fails
    ///
    /// # Pipilene
    /// 1. Convert input from [0, 1] to [-1, 1]: `x_norm = 2*x - 1`
    /// 2. Run model inference
    /// 3. Convert output from [-1, 1] to [0, 1]: `x = (x_norm + 1) / 2`
    /// 4. Clip output to [0, 1] range
    pub fn enhance(&mut self, face_image: &Array4<f32>) -> Result<Array4<f32>> {
        log_additional!(EVENT_ENHANCE_FACE, "Enhancing face");

        let shape = face_image.shape();
        if shape[0] != 1 || shape[1] != 3 || shape[2] != ENHANCE_SIZE || shape[3] != ENHANCE_SIZE {
            return Err(FaceSwapError::InvalidInput(format!(
                "Expected face image shape (1, 3, {}, {}), got ({}, {}, {}, {})",
                ENHANCE_SIZE, ENHANCE_SIZE, shape[0], shape[1], shape[2], shape[3]
            )));
        }

        // Preprocess: [0, 1] -> [-1, 1]
        // Formula: (x - 0.5) / 0.5 = 2*x - 1
        // Note: Model works with RGB directly (no BGR conversion needed, I believe)
        let normalized = face_image.mapv(|x| 2.0 * x - 1.0);

        let enhanced = {
            let outputs = self
                .session
                .run(inputs!["input" => TensorRef::from_array_view(normalized.view())?])
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

            let array_d = outputs["output"]
                .try_extract_array::<f32>()
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                .into_owned();

            array_d
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| FaceSwapError::ProcessingError(format!("{}", e)))?
        };

        // Postprocess: [-1, 1] -> [0, 1]
        // Formula is straightforward: (x + 1) / 2
        // + also clip to [0, 1] to handle any values outside range
        let denormalized = enhanced.mapv(|x| {
            let clipped = x.max(-1.0).min(1.0);
            (clipped + 1.0) / 2.0
        });

        Ok(denormalized)
    }
}
