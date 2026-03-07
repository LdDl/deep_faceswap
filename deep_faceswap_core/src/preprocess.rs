//! Image preprocessing utilities

use crate::error::Result;
use image::{DynamicImage, RgbImage};
use ndarray::Array4;

/// Load image from file
pub fn load_image(path: &str) -> Result<DynamicImage> {
    log::debug!("Loading image: {}", path);
    Ok(image::open(path)?)
}

/// Convert DynamicImage to RGB8
pub fn to_rgb8(img: &DynamicImage) -> RgbImage {
    img.to_rgb8()
}

/// Convert RgbImage to ndarray (NCHW format, normalized to [0, 1])
///
/// # Arguments
/// * `img` - Input RGB image
///
/// # Returns
/// Array with shape (1, 3, H, W) and values in [0.0, 1.0]
pub fn rgb_to_array(img: &RgbImage) -> Array4<f32> {
    let (width, height) = img.dimensions();
    let mut array = Array4::zeros((1, 3, height as usize, width as usize));

    for y in 0..height {
        for x in 0..width {
            let pixel = img.get_pixel(x, y);
            array[[0, 0, y as usize, x as usize]] = pixel[0] as f32 / 255.0;
            array[[0, 1, y as usize, x as usize]] = pixel[1] as f32 / 255.0;
            array[[0, 2, y as usize, x as usize]] = pixel[2] as f32 / 255.0;
        }
    }

    array
}

/// Convert ndarray (NCHW) to RgbImage
///
/// # Arguments
/// * `array` - Array with shape (1, 3, H, W) and values in [0.0, 1.0]
///
/// # Returns
/// RGB image
pub fn array_to_rgb(array: &Array4<f32>) -> Result<RgbImage> {
    let shape = array.shape();
    if shape[0] != 1 || shape[1] != 3 {
        return Err(crate::error::FaceSwapError::InvalidInput(
            format!("Expected shape (1, 3, H, W), got {:?}", shape)
        ));
    }

    let height = shape[2] as u32;
    let width = shape[3] as u32;
    let mut img = RgbImage::new(width, height);

    for y in 0..height {
        for x in 0..width {
            let r = (array[[0, 0, y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            let g = (array[[0, 1, y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            let b = (array[[0, 2, y as usize, x as usize]] * 255.0).clamp(0.0, 255.0) as u8;
            img.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    Ok(img)
}

/// Save image to file
pub fn save_image(img: &RgbImage, path: &str) -> Result<()> {
    log::info!("Saving image: {}", path);
    img.save(path)?;
    Ok(())
}
