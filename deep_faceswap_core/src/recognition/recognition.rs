//! Face recognition using ArcFace w600k_r50 from buffalo_l package
//!
//! Model: w600k_r50.onnx (ResNet50 trained on WebFace600K)
//!
//! This module extracts face embeddings used for identity preservation during face swapping.
//! The embeddings are 512-dimensional L2-normalized vectors that encode facial identity.

use crate::types::{FaceSwapError, Result};
use crate::utils::vector;
use crate::verbose::{get_verbose_level, VerboseLevel, EVENT_EXTRACT_EMBEDDING, EVENT_LOAD_MODEL};
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

const EMBEDDING_SIZE: usize = 512;

/// Face recognition model using ArcFace architecture
///
/// This recognizer extracts 512-dimensional face embeddings from aligned face images.
/// The model is based on ResNet50 trained on WebFace600K dataset.
///
/// # Model details
/// - Architecture: ResNet50 with ArcFace loss
/// - Input: 112x112 RGB face images (aligned, normalized to [0, 1])
/// - Output: 512-dimensional L2-normalized embedding vectors
/// - Dataset: WebFace600K
///
/// # Example
/// ```ignore
/// use deep_faceswap_core::recognition::FaceRecognizer;
///
/// let mut recognizer = FaceRecognizer::new("models/buffalo_l/w600k_r50.onnx")?;
/// let embedding = recognizer.extract_embedding(&aligned_face)?;
/// ```
pub struct FaceRecognizer {
    /// ONNX Runtime session for the face recognition model
    session: Session,
}

impl FaceRecognizer {
    /// Load face recognizer from ONNX model
    pub fn new(model_path: &str) -> Result<Self> {
        if !Path::new(model_path).exists() {
            return Err(FaceSwapError::ModelNotFound(model_path.to_string()));
        }

        log_main!(
            EVENT_LOAD_MODEL,
            "Loading face recognizer",
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

    /// Extract face embedding (512-d vector)
    ///
    /// # Arguments
    /// * `face_image` - Aligned face image as NCHW array (1, 3, 112, 112)
    ///
    /// # Returns
    /// 512-dimensional L2-normalized embedding
    pub fn extract_embedding(&mut self, face_image: &Array4<f32>) -> Result<Vec<f32>> {
        log_additional!(EVENT_EXTRACT_EMBEDDING, "Extracting face embedding");

        let shape = face_image.shape();
        if shape[0] != 1 || shape[1] != 3 || shape[2] != 112 || shape[3] != 112 {
            return Err(FaceSwapError::InvalidInput(format!(
                "Expected shape (1, 3, 112, 112), got ({}, {}, {}, {})",
                shape[0], shape[1], shape[2], shape[3]
            )));
        }

        let embedding = {
            let outputs = self
                .session
                .run(inputs!["input.1" => TensorRef::from_array_view(face_image.view())?])
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

            outputs["683"]
                .try_extract_array::<f32>()
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                .into_owned()
        };

        let mut embedding_vec: Vec<f32> = embedding.iter().copied().collect();

        if embedding_vec.len() != EMBEDDING_SIZE {
            return Err(FaceSwapError::ProcessingError(format!(
                "Expected {} embedding dimensions, got {}",
                EMBEDDING_SIZE,
                embedding_vec.len()
            )));
        }

        vector::l2_normalize(&mut embedding_vec);

        Ok(embedding_vec)
    }
}

