//! Face detection using YOLOv8n from buffalo_l package
//!
//! Model: det_10g.onnx (YOLOv8n-based detector)

use crate::error::Result;
use ndarray::Array4;

/// Bounding box in [x1, y1, x2, y2] format
#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

/// Detected face with bounding box and landmarks
#[derive(Debug, Clone)]
pub struct DetectedFace {
    pub bbox: BBox,
    // 5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth
    pub landmarks: [[f32; 2]; 5],
    pub det_score: f32,
}

/// Face detector using YOLOv8n
pub struct FaceDetector {
    model_path: String,
    // TODO: Add ort session
}

impl FaceDetector {
    /// Load face detector from ONNX model
    pub fn new(model_path: &str) -> Result<Self> {
        log::info!("Loading face detector: {}", model_path);
        // TODO: Load tract model
        Ok(Self {
            model_path: model_path.to_string(),
        })
    }

    /// Detect faces in image
    ///
    /// # Arguments
    /// * `image` - Input image as NCHW array (1, 3, H, W)
    ///
    /// # Returns
    /// Vector of detected faces sorted by score (descending)
    pub fn detect(&self, _image: &Array4<f32>) -> Result<Vec<DetectedFace>> {
        log::debug!("Detecting faces...");
        // TODO: Implement detection
        Ok(vec![])
    }
}
