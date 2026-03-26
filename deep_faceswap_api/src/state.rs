//! Application state and model management

use crate::jobs::JobState;
use deep_faceswap_core::detection::FaceDetector;
use deep_faceswap_core::enhancer::FaceEnhancer;
use deep_faceswap_core::landmark::LandmarkDetector;
use deep_faceswap_core::recognition::FaceRecognizer;
use deep_faceswap_core::swapper::FaceSwapper;
use std::collections::HashMap;
use std::sync::Mutex;

/// All ML models loaded once at startup
pub struct ModelRegistry {
    pub detector: Mutex<FaceDetector>,
    pub recognizer: Mutex<FaceRecognizer>,
    pub swapper: Mutex<FaceSwapper>,
    pub enhancer: Mutex<Option<FaceEnhancer>>,
    pub landmark_detector: Mutex<Option<LandmarkDetector>>,
}

/// Shared application state passed to all route handlers
pub struct AppState {
    pub models: ModelRegistry,
    pub jobs: Mutex<HashMap<String, JobState>>,
    pub allowed_dirs: Vec<String>,
    pub tmp_dir: String,
    /// Maps session_id -> base tmp_dir used for that session
    pub session_dirs: Mutex<HashMap<String, String>>,
}

impl AppState {
    pub fn new(
        detector: FaceDetector,
        recognizer: FaceRecognizer,
        swapper: FaceSwapper,
        enhancer: Option<FaceEnhancer>,
        landmark_detector: Option<LandmarkDetector>,
        allowed_dirs: Vec<String>,
        tmp_dir: String,
    ) -> Self {
        Self {
            models: ModelRegistry {
                detector: Mutex::new(detector),
                recognizer: Mutex::new(recognizer),
                swapper: Mutex::new(swapper),
                enhancer: Mutex::new(enhancer),
                landmark_detector: Mutex::new(landmark_detector),
            },
            jobs: Mutex::new(HashMap::new()),
            allowed_dirs,
            tmp_dir,
            session_dirs: Mutex::new(HashMap::new()),
        }
    }

    /// Check if a path is under one of the allowed directories
    pub fn is_path_allowed(&self, path: &str) -> bool {
        if self.allowed_dirs.is_empty() {
            return true;
        }
        let canonical = match std::fs::canonicalize(path) {
            Ok(p) => p,
            Err(_) => return false,
        };
        let canonical_str = canonical.to_string_lossy();
        self.allowed_dirs
            .iter()
            .any(|dir| canonical_str.starts_with(dir))
    }
}
