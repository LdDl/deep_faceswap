//! Multi-face selection and mapping
//!
//! This module handles interactive face selection when multiple faces are detected
//! in source or target images. It provides:
//! - Saving face crops for visual inspection
//! - Interactive CLI prompts for face mapping
//! - Building face mapping lists based on user input

pub mod prompt;
pub mod selection;

pub use selection::build_face_mappings;
