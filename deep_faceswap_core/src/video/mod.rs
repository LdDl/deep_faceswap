//! Video processing module

pub mod analysis;
pub mod encoding;
pub mod extraction;

pub use analysis::*;
pub use encoding::encode_video;
pub use extraction::extract_frames;
