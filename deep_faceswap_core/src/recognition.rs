//! Face recognition using ArcFace w600k_r50 from buffalo_l package
//!
//! Model: w600k_r50.onnx (ResNet50 trained on WebFace600K)

use crate::error::Result;
use ndarray::Array4;

/// Face recognition model (ArcFace)
pub struct FaceRecognizer {
    model_path: String,
    // TODO: Add ort session
}

impl FaceRecognizer {
    /// Load face recognizer from ONNX model
    pub fn new(model_path: &str) -> Result<Self> {
        log::info!("Loading face recognizer: {}", model_path);
        // TODO: Load tract model
        Ok(Self {
            model_path: model_path.to_string(),
        })
    }

    /// Extract face embedding (512-d vector)
    ///
    /// # Arguments
    /// * `face_image` - Aligned face image as NCHW array (1, 3, 112, 112)
    ///
    /// # Returns
    /// 512-dimensional L2-normalized embedding
    pub fn extract_embedding(&self, _face_image: &Array4<f32>) -> Result<Vec<f32>> {
        log::debug!("Extracting face embedding...");
        // TODO: Implement embedding extraction
        Ok(vec![0.0; 512])
    }
}
