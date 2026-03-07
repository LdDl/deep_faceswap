//! Deep FaceSwap Core Library
//!
//! Basic face swap between two images (single source + single target)
//!
//! This library provides:
//! - Face detection (YOLOv8n from buffalo_l)
//! - Face recognition (ArcFace w600k_r50 from buffalo_l)
//! - Face swapping (inswapper_128)

pub mod types;

pub mod alignment;
pub mod detection;
pub mod recognition;
pub mod swap;
pub mod swapper;
pub mod utils;

#[macro_use]
pub mod verbose;

pub use swap::swap_faces;
pub use types::{FaceSwapError, Result};
