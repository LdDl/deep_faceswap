//! Image I/O utilities

use crate::types::Result;
use crate::verbose::{EVENT_LOAD_IMAGE, EVENT_SAVE_IMAGE};
use crate::{log_additional, log_all, log_main};
use image::{DynamicImage, RgbImage};

/// Load image from file
pub fn load_image(path: &str) -> Result<DynamicImage> {
    log_additional!(EVENT_LOAD_IMAGE, "Loading image", path = path);
    Ok(image::open(path)?)
}

/// Convert DynamicImage to RGB8
pub fn to_rgb8(img: &DynamicImage) -> RgbImage {
    img.to_rgb8()
}

/// Save image to file
/// Purpose: for image processing when a video is not involved
pub fn save_image(img: &RgbImage, path: &str) -> Result<()> {
    log_main!(EVENT_SAVE_IMAGE, "Saving image", path = path);
    img.save(path)?;
    Ok(())
}

/// Save image to file quietly
/// Main purpose: for video frame processing, logs at trace level
pub fn save_image_quiet(img: &RgbImage, path: &str) -> Result<()> {
    log_all!(EVENT_SAVE_IMAGE, "Saving image", path = path);
    img.save(path)?;
    Ok(())
}
