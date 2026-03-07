use std::fmt;

#[derive(Debug)]
pub enum FaceSwapError {
    Io(std::io::Error),
    Image(image::ImageError),
    Ort(String),
    NoFacesDetected,
    MultipleFacesDetected(usize),
    ModelNotFound(String),
    InvalidInput(String),
    NotImplemented(String),
    ProcessingError(String),
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

pub type Result<T> = std::result::Result<T, FaceSwapError>;
