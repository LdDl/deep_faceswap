//! Face swapping using inswapper_128
//!
//! This module implements face swapping using the inswapper_128 model, which takes
//! an aligned target face and a source face embedding to generate a swapped face
//! with the target's pose/expression and the source's identity.
//!
//! # Model details
//! - Model: inswapper_128.onnx
//! - Input size: 128x128 aligned face images
//! - Source: 512-dimensional face embedding (from ArcFace)
//! - Target: Aligned face image in NCHW format
//! - Output: Swapped face in NCHW format (128x128)
//!
//! # Emap transformation
//! The model requires an emap (embedding map) matrix (512x512) that transforms
//! the source embedding into the model's latent space. This matrix must be extracted
//! from the ONNX model file separately and saved as a binary file.

use crate::types::{FaceSwapError, Result};
use crate::verbose::{get_verbose_level, VerboseLevel, EVENT_LOAD_MODEL, EVENT_SWAP_FACE};
use crate::{log_additional, log_main};
use ndarray::{Array1, Array2, Array4};
use ort::{
    inputs,
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::path::Path;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

const SWAP_SIZE: usize = 128;

/// Face swapper model using inswapper architecture
///
/// This swapper combines a target face image with a source face embedding to produce
/// a swapped face that maintains the target's pose/expression while adopting the
/// source's identity features.
///
/// # Implementation details
/// - Uses ONNX Runtime for model inference
/// - Requires 128x128 aligned face images
/// - Source embedding is transformed via emap matrix before swapping
/// - Output is normalized and ready for paste-back to original image
///
/// # Example
/// ```ignore
/// use deep_faceswap_core::swapper::FaceSwapper;
///
/// let mut swapper = FaceSwapper::new("models/inswapper_128.onnx")?;
/// let swapped = swapper.swap(&target_face, &source_embedding)?;
/// ```
pub struct FaceSwapper {
    /// ONNX Runtime session for the face swapping model
    session: Session,
    /// Emap transformation matrix (512x512) for embedding projection
    emap: Array2<f32>,
}

impl FaceSwapper {
    /// Load face swapper from ONNX model
    ///
    /// Loads the inswapper_128 model and its emap transformation matrix.
    /// The emap file must exist alongside the model with `_emap.bin` suffix.
    ///
    /// # Arguments
    /// * `model_path` - Path to inswapper_128.onnx model file
    ///
    /// # Returns
    /// Initialized face swapper ready for inference
    ///
    /// # Errors
    /// - `ModelNotFound` if model file or emap file doesn't exist
    /// - `Ort` if ONNX Runtime initialization fails
    /// - `ProcessingError` if emap file has wrong size
    ///
    /// # Required files
    /// - `inswapper_128.onnx` - Main model file
    /// - `inswapper_128_emap.bin` - Emap matrix (512x512 float32)
    pub fn new(model_path: &str) -> Result<Self> {
        if !Path::new(model_path).exists() {
            return Err(FaceSwapError::ModelNotFound(model_path.to_string()));
        }

        log_main!(EVENT_LOAD_MODEL, "Loading face swapper", path = model_path);

        // Load emap from ONNX model file
        let emap = Self::load_emap(model_path)?;

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

        Ok(Self { session, emap })
    }

    /// Load emap transformation matrix from binary file
    ///
    /// The emap matrix projects face embeddings into the model's latent space.
    /// It must be extracted from the ONNX model file once and saved as binary.
    ///
    /// # Arguments
    /// * `model_path` - Path to the ONNX model (emap path is derived from this)
    ///
    /// # Returns
    /// 512x512 transformation matrix as Array2<f32>
    ///
    /// # Errors
    /// - `ModelNotFound` if emap file doesn't exist
    /// - `ProcessingError` if file size is incorrect or parsing fails
    ///
    /// # Emap extraction
    /// The emap file should be extracted from the ONNX model using Python:
    /// ```python
    /// import onnx
    /// from onnx import numpy_helper
    /// model = onnx.load('inswapper_128.onnx')
    /// emap = numpy_helper.to_array(model.graph.initializer[-1])
    /// emap.astype(np.float32).tofile('inswapper_128_emap.bin')
    /// ```
    fn load_emap(model_path: &str) -> Result<Array2<f32>> {
        use std::fs::File;
        use std::io::Read;

        // Derive emap path from model path
        let emap_path = model_path.replace(".onnx", "_emap.bin");

        let mut file = File::open(&emap_path).map_err(|e| {
            FaceSwapError::ModelNotFound(format!(
                "Cannot open emap file {}: {}. Please extract emap using Python.",
                emap_path, e
            ))
        })?;

        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)
            .map_err(|e| FaceSwapError::ProcessingError(format!("Cannot read emap file: {}", e)))?;

        // Parse as f32 array
        let float_data: Vec<f32> = buffer
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        // emap should be 512x512
        if float_data.len() != 512 * 512 {
            return Err(FaceSwapError::ProcessingError(format!(
                "Expected emap to be 512x512, got {} elements",
                float_data.len()
            )));
        }

        Array2::from_shape_vec((512, 512), float_data)
            .map_err(|e| FaceSwapError::ProcessingError(format!("Cannot create emap array: {}", e)))
    }

    /// Swap face in target image with source embedding
    ///
    /// Takes an aligned target face and source embedding to produce a swapped face.
    /// The source embedding is transformed via emap matrix before being passed to
    /// the model along with the target face.
    ///
    /// # Arguments
    /// * `target_face_aligned` - Aligned target face as NCHW array (1, 3, 128, 128), normalized to [0, 1]
    /// * `source_embedding` - 512-dimensional L2-normalized embedding from source face
    ///
    /// # Returns
    /// Swapped face as NCHW array (1, 3, 128, 128), normalized to [0, 1]
    ///
    /// # Errors
    /// - `InvalidInput` if target face shape is not (1, 3, 128, 128)
    /// - `InvalidInput` if source embedding length is not 512
    /// - `Ort` if model inference fails
    /// - `ProcessingError` if output shape conversion fails
    ///
    /// # Process
    /// 1. Transform source embedding: `latent = embedding @ emap`
    /// 2. Normalize latent vector: `latent = latent / ||latent||`
    /// 3. Run model inference with target face and transformed embedding
    /// 4. Return swapped face in same format as target
    pub fn swap(
        &mut self,
        target_face_aligned: &Array4<f32>,
        source_embedding: &[f32],
    ) -> Result<Array4<f32>> {
        log_additional!(EVENT_SWAP_FACE, "Swapping face");

        let shape = target_face_aligned.shape();
        if shape[0] != 1 || shape[1] != 3 || shape[2] != SWAP_SIZE || shape[3] != SWAP_SIZE {
            return Err(FaceSwapError::InvalidInput(format!(
                "Expected target face shape (1, 3, {}, {}), got ({}, {}, {}, {})",
                SWAP_SIZE, SWAP_SIZE, shape[0], shape[1], shape[2], shape[3]
            )));
        }

        if source_embedding.len() != 512 {
            return Err(FaceSwapError::InvalidInput(format!(
                "Expected 512-d source embedding, got {}",
                source_embedding.len()
            )));
        }

        // Transform source embedding with emap matrix
        // latent = embedding @ emap
        // latent = latent / norm(latent)
        let embedding = Array1::from_vec(source_embedding.to_vec());
        let embedding_2d = embedding.insert_axis(ndarray::Axis(0));

        let transformed = embedding_2d.dot(&self.emap);
        let norm: f32 = transformed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let normalized = transformed / norm;

        let swapped = {
            let outputs = self
                .session
                .run(inputs![
                    "target" => TensorRef::from_array_view(target_face_aligned.view())?,
                    "source" => TensorRef::from_array_view(normalized.view())?
                ])
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

            let array_d = outputs["output"]
                .try_extract_array::<f32>()
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                .into_owned();

            array_d
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(|e| FaceSwapError::ProcessingError(format!("{}", e)))?
        };

        Ok(swapped)
    }
}
