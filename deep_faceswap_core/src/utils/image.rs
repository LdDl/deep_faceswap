//! Image I/O utilities

use crate::types::Result;
use crate::verbose::{EVENT_LOAD_IMAGE, EVENT_SAVE_IMAGE};
use crate::{log_additional, log_main};
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
pub fn save_image(img: &RgbImage, path: &str) -> Result<()> {
    log_main!(EVENT_SAVE_IMAGE, "Saving image", path = path);
    img.save(path)?;
    Ok(())
}
