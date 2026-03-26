//! RGB format conversion utilities
//!
//! This module provides conversions between image::RgbImage and ndarray formats.
//! Two types of conversions are available:
//! - Array4<f32> (NCHW, normalized [0.0, 1.0]) for neural network inputs
//! - Array3<u8> (HWC, raw [0, 255]) for image processing operations

use crate::types::{FaceSwapError, Result};
use image::RgbImage;
use ndarray::{Array3, Array4};

/// Convert RgbImage to Array4<f32> (NCHW format, normalized to [0.0, 1.0])
///
/// This conversion creates a normalized ndarray tensor suitable for neural network input.
/// The output uses NCHW (batch, channels, height, width) format with pixel values
/// normalized to the range [0.0, 1.0].
///
/// # Arguments
/// * `img` - Input RGB image
///
/// # Returns
/// Array with shape (1, 3, H, W) and values in [0.0, 1.0]
///
/// # Example
/// ```rust
/// use deep_faceswap_core::utils::rgb::rgb_to_array;
/// use image::RgbImage;
///
/// let img = RgbImage::new(112, 112);
/// let array = rgb_to_array(&img);
/// assert_eq!(array.shape(), &[1, 3, 112, 112]);
/// ```
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

/// Convert Array4<f32> (NCHW format) to RgbImage
///
/// This conversion creates an RgbImage from a normalized ndarray tensor.
/// The input should use NCHW (batch, channels, height, width) format with
/// pixel values in the range [0.0, 1.0]. Values are clamped and scaled to [0, 255].
///
/// # Arguments
/// * `array` - Array with shape (1, 3, H, W) and values in [0.0, 1.0]
///
/// # Returns
/// RGB image, or error if array shape is invalid
///
/// # Errors
/// Returns `FaceSwapError::InvalidInput` if the array doesn't have shape (1, 3, H, W)
///
/// # Example
/// ```rust
/// use deep_faceswap_core::utils::rgb::array_to_rgb;
/// use ndarray::Array4;
///
/// let array = Array4::<f32>::zeros((1, 3, 128, 128));
/// let img = array_to_rgb(&array).unwrap();
/// assert_eq!(img.dimensions(), (128, 128));
/// ```
pub fn array_to_rgb(array: &Array4<f32>) -> Result<RgbImage> {
    let shape = array.shape();
    if shape[0] != 1 || shape[1] != 3 {
        return Err(FaceSwapError::InvalidInput(format!(
            "Expected shape (1, 3, H, W), got {:?}",
            shape
        )));
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

/// Convert RgbImage to Array3<u8> (HWC format, raw bytes)
///
/// This conversion creates an ndarray with raw pixel values for image processing.
/// Unlike `rgb_to_array`, this keeps values in [0, 255] range without normalization.
///
/// # Arguments
/// * `img` - Input RGB image
///
/// # Returns
/// Array with shape (H, W, 3) and values in [0, 255]
///
/// # Example
/// ```rust
/// use deep_faceswap_core::utils::rgb::rgb_to_array3;
/// use image::RgbImage;
///
/// let img = RgbImage::new(10, 10);
/// let array = rgb_to_array3(&img);
/// assert_eq!(array.shape(), &[10, 10, 3]);
/// ```
pub fn rgb_to_array3(img: &RgbImage) -> Array3<u8> {
    let (width, height) = img.dimensions();

    // RgbImage stores pixels as [R,G,B,R,G,B,...] in row-major order.
    // Array3<u8> with shape (H,W,3) in standard layout has identical memory layout.
    // Single bulk copy instead of H*W individual get_pixel calls.
    Array3::from_shape_vec((height as usize, width as usize, 3), img.as_raw().clone())
        .expect("RgbImage raw buffer size must match H*W*3")
}

/// Convert Array3<u8> (HWC format) to RgbImage
///
/// This conversion creates an RgbImage from raw pixel values.
/// Unlike `array_to_rgb`, this works with Array3 (HWC) instead of Array4 (NCHW),
/// and expects raw values in [0, 255] range.
///
/// # Arguments
/// * `array` - Array with shape (H, W, 3) and values in [0, 255]
///
/// # Returns
/// RGB image
///
/// # Example
/// ```rust
/// use deep_faceswap_core::utils::rgb::array3_to_rgb;
/// use ndarray::Array3;
///
/// let array = Array3::<u8>::zeros((10, 10, 3));
/// let img = array3_to_rgb(&array);
/// assert_eq!(img.dimensions(), (10, 10));
/// ```
pub fn array3_to_rgb(array: &Array3<u8>) -> RgbImage {
    let (h, w, _) = (array.shape()[0], array.shape()[1], array.shape()[2]);

    // If the array is contiguous in standard (row-major C) layout, use bulk copy.
    // Otherwise fall back to element-by-element collection.
    let data = if array.is_standard_layout() {
        array.as_slice().unwrap().to_vec()
    } else {
        array.iter().cloned().collect()
    };

    RgbImage::from_raw(w as u32, h as u32, data).expect("Array3 data size must match W*H*3")
}
