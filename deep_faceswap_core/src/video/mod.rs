//! Video processing module

pub mod encoding;
pub mod extraction;

pub use encoding::encode_video;
pub use extraction::extract_frames;
