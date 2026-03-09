//! Core types for face swapping

use std::fmt;

/// Custom error type for face swap operations
#[derive(Debug)]
pub enum FaceSwapError {
    /// File system I/O error (reading/writing files)
    Io(std::io::Error),
    /// Image loading or decoding error
    Image(image::ImageError),
    /// ONNX Runtime inference error
    Ort(String),
    /// No faces detected in the input image
    NoFacesDetected,
    /// Multiple faces detected when expecting exactly one
    MultipleFacesDetected(usize),
    /// Required model file not found at specified path
    ModelNotFound(String),
    /// Invalid input parameters or data format
    InvalidInput(String),
    /// Requested feature not yet implemented
    NotImplemented(String),
    /// Error during image processing or transformation
    ProcessingError(String),
    /// Invalid face mapping provided (indices out of range)
    InvalidMapping(String),
    /// User cancelled interactive selection
    UserCancelled,
}

impl fmt::Display for FaceSwapError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "IO error: {}", e),
            Self::Image(e) => write!(f, "Image error: {}", e),
            Self::Ort(e) => write!(f, "ONNX Runtime error: {}", e),
            Self::NoFacesDetected => write!(f, "No faces detected in image"),
            Self::MultipleFacesDetected(n) => {
                write!(f, "Multiple faces detected (expected 1, found {})", n)
            }
            Self::ModelNotFound(path) => write!(f, "Model not found: {}", path),
            Self::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
            Self::NotImplemented(feature) => write!(f, "Feature not implemented: {}", feature),
            Self::ProcessingError(msg) => write!(f, "Processing error: {}", msg),
            Self::InvalidMapping(msg) => write!(f, "Invalid face mapping: {}", msg),
            Self::UserCancelled => write!(f, "Operation cancelled by user"),
        }
    }
}

impl std::error::Error for FaceSwapError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            Self::Image(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for FaceSwapError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

impl From<image::ImageError> for FaceSwapError {
    fn from(e: image::ImageError) -> Self {
        Self::Image(e)
    }
}

impl From<ort::Error> for FaceSwapError {
    fn from(e: ort::Error) -> Self {
        Self::Ort(format!("{}", e))
    }
}

pub type Result<T> = std::result::Result<T, FaceSwapError>;

/// Bounding box in [x1, y1, x2, y2] format
#[derive(Debug, Clone)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
}

/// Detected face from face detection model
///
/// Contains bounding box, facial landmarks, and detection confidence score.
/// The landmarks follow the 5-point format used by InsightFace models.
#[derive(Debug, Clone)]
pub struct DetectedFace {
    /// Face bounding box in original image coordinates
    pub bbox: BBox,

    /// 5 facial keypoints in [x, y] format
    /// Order: left_eye, right_eye, nose, left_mouth_corner, right_mouth_corner
    /// Coordinates are in original image space before any alignment
    pub landmarks: [[f32; 2]; 5],

    /// Detection confidence score in range [0.0, 1.0]
    /// Higher values indicate more confident detections
    pub det_score: f32,
}

/// Face aligned to canonical pose with transformation matrix
///
/// Contains the original detected face, the aligned face image ready for
/// recognition or swapping, and the similarity transformation matrix used
/// for alignment. The transformation can be used later to paste the swapped
/// face back to the original image position.
#[derive(Debug, Clone)]
pub struct AlignedFace {
    /// Original detected face metadata
    pub face: DetectedFace,

    /// Aligned face image in NCHW format (1, 3, size, size)
    /// Normalized to [0.0, 1.0] range, RGB channel order
    /// Size is typically 112 for recognition or 128 for swapping
    pub aligned_image: ndarray::Array4<f32>,

    /// Similarity transformation matrix (2x3) used for alignment
    /// Maps from original image coordinates to aligned face coordinates
    /// Form: [[a, -b, tx], [b, a, ty]] where a = scale*cos(theta), b = scale*sin(theta)
    /// This is the forward transform (original -> aligned)
    pub transform: ndarray::Array2<f32>,
}

/// Face crop information for multi-face selection
///
/// Contains a detected face and the path where its cropped image was saved.
/// Used during interactive face selection to let users visually identify faces.
#[derive(Debug, Clone)]
pub struct FaceCropInfo {
    /// The detected face with bbox and landmarks
    pub face: DetectedFace,

    /// Path to saved crop image (e.g., "./tmp/face_crops/source/face_0.jpg")
    pub crop_path: String,

    /// Face index in detection results (0-based)
    pub index: usize,
}

/// Mapping from source face to target face for multi-face swapping
///
/// Specifies which source face should be swapped onto which target face.
/// Indices correspond to detection order (sorted by score, descending).
#[derive(Debug, Clone)]
pub struct FaceMapping {
    /// Index of source face (0-based)
    pub source_idx: usize,

    /// Index of target face (0-based)
    pub target_idx: usize,
}

/// Source face with metadata about its origin
///
/// When multiple source images are provided, this tracks which image
/// each face came from, along with the face data itself.
#[derive(Debug, Clone)]
pub struct SourceFaceInfo {
    /// The detected face with bbox and landmarks
    pub face: DetectedFace,

    /// Index of source image this face was detected in (0-based)
    pub source_image_index: usize,

    /// Filename of source image (e.g., "image1.jpg")
    pub source_filename: String,
}
