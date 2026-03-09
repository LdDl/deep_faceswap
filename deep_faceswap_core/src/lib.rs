//! Deep FaceSwap Core Library
//!
//! Basic face swap between two images (single source + single target)
//!
//! This library provides:
//! - Face detection (YOLOv8n from buffalo_l)
//! - Face recognition (ArcFace w600k_r50 from buffalo_l)
//! - Face swapping (inswapper_128)
//! - Face enhancement (GFPGAN)

pub mod types;

pub mod alignment;
pub mod detection;
pub mod enhancer;
pub mod landmark;
pub mod mouth_mask;
pub mod multi_face;
pub mod recognition;
pub mod swap;
pub mod swapper;
pub mod utils;
pub mod video;

#[macro_use]
pub mod verbose;

pub use swap::{swap_faces, swap_video};
pub use types::{FaceSwapError, Result};
