//! Color space conversion and color transfer utilities
//!
//! Provides RGB <-> CIE LAB conversion and LAB-based color transfer.
//! Used in mouth mask blending to match the original mouth colors
//! with the swapped face lighting.

use ndarray::Array3;

/// Convert sRGB gamma-encoded value to linear RGB.
///
/// Applies the inverse sRGB companding function.
/// Input and output are in [0.0, 1.0] range.
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::color::srgb_to_linear;
///
/// assert!((srgb_to_linear(0.0) - 0.0).abs() < 1e-6);
/// assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-6);
/// // Mid-gray is darker in linear space
/// assert!(srgb_to_linear(0.5) < 0.5);
/// ```
pub fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Convert linear RGB value to sRGB gamma-encoded.
///
/// Applies the sRGB companding function.
/// Input and output are in [0.0, 1.0] range.
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::color::linear_to_srgb;
///
/// assert!((linear_to_srgb(0.0) - 0.0).abs() < 1e-6);
/// assert!((linear_to_srgb(1.0) - 1.0).abs() < 1e-6);
/// ```
pub fn linear_to_srgb(c: f32) -> f32 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// CIE LAB forward transfer function.
///
/// Maps linear light values through the CIE LAB perceptual
/// curve. Values below the threshold use a linear segment
/// to avoid numerical issues near zero.
///
/// # Arguments
/// * `t` - Linear light value (X/Xn, Y/Yn, or Z/Zn ratio)
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::color::lab_f;
///
/// // Above threshold: cube root
/// assert!((lab_f(1.0) - 1.0).abs() < 1e-6);
/// // Below threshold: linear approximation
/// assert!(lab_f(0.001) > 0.0);
/// ```
pub fn lab_f(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta * delta * delta {
        t.cbrt()
    } else {
        t / (3.0 * delta * delta) + 4.0 / 29.0
    }
}

/// CIE LAB inverse transfer function.
///
/// Inverse of [`lab_f`]. Maps perceptual LAB values back
/// to linear light ratios.
///
/// # Arguments
/// * `t` - Perceptual value from LAB computation
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::color::{lab_f, lab_f_inv};
///
/// // Roundtrip
/// let val = 0.5;
/// assert!((lab_f_inv(lab_f(val)) - val).abs() < 1e-5);
/// ```
pub fn lab_f_inv(t: f32) -> f32 {
    let delta: f32 = 6.0 / 29.0;
    if t > delta {
        t * t * t
    } else {
        3.0 * delta * delta * (t - 4.0 / 29.0)
    }
}

// D65 white point
const D65_X: f32 = 0.95047;
const D65_Y: f32 = 1.0;
const D65_Z: f32 = 1.08883;

/// Convert a single RGB pixel (0-255) to CIE LAB color space.
///
/// Uses D65 illuminant and sRGB color space.
/// L is in [0, 100], a and b are typically in [-128, 127].
///
/// # Arguments
/// * `r`, `g`, `b` - RGB values in 0-255 range
///
/// # Returns
/// Tuple (L, a, b) in CIE LAB space
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::color::rgb_to_lab;
///
/// // Black has L=0
/// let (l, _, _) = rgb_to_lab(0, 0, 0);
/// assert!(l.abs() < 0.1);
///
/// // White has L=100
/// let (l, _, _) = rgb_to_lab(255, 255, 255);
/// assert!((l - 100.0).abs() < 0.1);
///
/// // Pure red has positive a (red-green axis)
/// let (_, a, _) = rgb_to_lab(255, 0, 0);
/// assert!(a > 0.0);
/// ```
pub fn rgb_to_lab(r: u8, g: u8, b: u8) -> (f32, f32, f32) {
    let rf = srgb_to_linear(r as f32 / 255.0);
    let gf = srgb_to_linear(g as f32 / 255.0);
    let bf = srgb_to_linear(b as f32 / 255.0);

    // sRGB to XYZ (D65)
    let x = 0.4124564 * rf + 0.3575761 * gf + 0.1804375 * bf;
    let y = 0.2126729 * rf + 0.7151522 * gf + 0.0721750 * bf;
    let z = 0.0193339 * rf + 0.1191920 * gf + 0.9503041 * bf;

    // XYZ to Lab
    let fx = lab_f(x / D65_X);
    let fy = lab_f(y / D65_Y);
    let fz = lab_f(z / D65_Z);

    let l = 116.0 * fy - 16.0;
    let a = 500.0 * (fx - fy);
    let b_val = 200.0 * (fy - fz);

    (l, a, b_val)
}

/// Convert a CIE LAB color to RGB (0-255).
///
/// Uses D65 illuminant and sRGB color space.
/// Out-of-gamut values are clamped to [0, 255].
///
/// # Arguments
/// * `l` - Lightness in [0, 100]
/// * `a` - Green-red axis
/// * `b_val` - Blue-yellow axis
///
/// # Returns
/// Tuple (r, g, b) in 0-255 range
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::color::{rgb_to_lab, lab_to_rgb};
///
/// // Roundtrip test
/// let (l, a, b) = rgb_to_lab(200, 100, 50);
/// let (r, g, bv) = lab_to_rgb(l, a, b);
/// assert!((r as i16 - 200).abs() <= 1);
/// assert!((g as i16 - 100).abs() <= 1);
/// assert!((bv as i16 - 50).abs() <= 1);
/// ```
pub fn lab_to_rgb(l: f32, a: f32, b_val: f32) -> (u8, u8, u8) {
    let fy = (l + 16.0) / 116.0;
    let fx = a / 500.0 + fy;
    let fz = fy - b_val / 200.0;

    let x = D65_X * lab_f_inv(fx);
    let y = D65_Y * lab_f_inv(fy);
    let z = D65_Z * lab_f_inv(fz);

    // XYZ to sRGB
    let rf = 3.2404542 * x - 1.5371385 * y - 0.4985314 * z;
    let gf = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z;
    let bf = 0.0556434 * x - 0.2040259 * y + 1.0572252 * z;

    let r = (linear_to_srgb(rf.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
    let g = (linear_to_srgb(gf.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;
    let b = (linear_to_srgb(bf.clamp(0.0, 1.0)) * 255.0 + 0.5) as u8;

    (r, g, b)
}

/// Transfer color statistics from target to source using LAB color space.
///
/// Adjusts the source image colors so that its mean and standard deviation
/// in LAB space match the target image. This is useful for blending regions
/// from different images with different lighting.
///
/// Both images must be `Array3<u8>` in HWC format with 3 RGB channels.
/// The images can have different dimensions.
///
/// # Algorithm
/// For each LAB channel independently:
/// `result = (source - source_mean) * (target_std / source_std) + target_mean`
///
/// # Arguments
/// * `source` - Source image whose colors will be adjusted (HWC, u8, RGB)
/// * `target` - Target image whose color statistics to match (HWC, u8, RGB)
///
/// # Returns
/// Color-corrected source image with same dimensions as source
///
/// # Example
/// ```
/// use ndarray::Array3;
/// use deep_faceswap_core::utils::color::color_transfer_lab;
///
/// let source = Array3::<u8>::from_elem((10, 10, 3), 128);
/// let target = Array3::<u8>::from_elem((10, 10, 3), 200);
///
/// let result = color_transfer_lab(&source, &target);
/// // Result should be shifted toward target's brightness
/// assert!(result[[5, 5, 0]] > 128);
/// ```
pub fn color_transfer_lab(source: &Array3<u8>, target: &Array3<u8>) -> Array3<u8> {
    let (sh, sw, _) = source.dim();
    let (th, tw, _) = target.dim();

    let src_pixels = sh * sw;
    let tgt_pixels = th * tw;

    let mut src_l = vec![0.0f32; src_pixels];
    let mut src_a = vec![0.0f32; src_pixels];
    let mut src_b = vec![0.0f32; src_pixels];

    let mut tgt_l = vec![0.0f32; tgt_pixels];
    let mut tgt_a = vec![0.0f32; tgt_pixels];
    let mut tgt_b = vec![0.0f32; tgt_pixels];

    for y in 0..sh {
        for x in 0..sw {
            let i = y * sw + x;
            let (l, a, b) = rgb_to_lab(source[[y, x, 0]], source[[y, x, 1]], source[[y, x, 2]]);
            src_l[i] = l;
            src_a[i] = a;
            src_b[i] = b;
        }
    }

    for y in 0..th {
        for x in 0..tw {
            let i = y * tw + x;
            let (l, a, b) = rgb_to_lab(target[[y, x, 0]], target[[y, x, 1]], target[[y, x, 2]]);
            tgt_l[i] = l;
            tgt_a[i] = a;
            tgt_b[i] = b;
        }
    }

    let (src_mean_l, src_std_l) = mean_std(&src_l);
    let (src_mean_a, src_std_a) = mean_std(&src_a);
    let (src_mean_b, src_std_b) = mean_std(&src_b);

    let (tgt_mean_l, tgt_std_l) = mean_std(&tgt_l);
    let (tgt_mean_a, tgt_std_a) = mean_std(&tgt_a);
    let (tgt_mean_b, tgt_std_b) = mean_std(&tgt_b);

    let eps = 1e-6;

    let mut result = Array3::<u8>::zeros((sh, sw, 3));
    for y in 0..sh {
        for x in 0..sw {
            let i = y * sw + x;

            let l = (src_l[i] - src_mean_l) * (tgt_std_l / (src_std_l + eps)) + tgt_mean_l;
            let a = (src_a[i] - src_mean_a) * (tgt_std_a / (src_std_a + eps)) + tgt_mean_a;
            let b = (src_b[i] - src_mean_b) * (tgt_std_b / (src_std_b + eps)) + tgt_mean_b;

            let (r, g, bv) = lab_to_rgb(l, a, b);
            result[[y, x, 0]] = r;
            result[[y, x, 1]] = g;
            result[[y, x, 2]] = bv;
        }
    }

    result
}

/// Compute mean and standard deviation of a float slice.
///
/// Uses population standard deviation (divides by N, not N-1).
///
/// # Arguments
/// * `values` - Slice of f32 values (must not be empty)
///
/// # Returns
/// Tuple (mean, std_dev)
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::color::mean_std;
///
/// let values = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let (mean, std) = mean_std(&values);
/// assert!((mean - 5.0).abs() < 1e-6);
/// assert!((std - 2.0).abs() < 1e-6);
/// ```
pub fn mean_std(values: &[f32]) -> (f32, f32) {
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, variance.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb_lab_roundtrip() {
        let test_colors: Vec<(u8, u8, u8)> = vec![
            (0, 0, 0),
            (255, 255, 255),
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (128, 128, 128),
            (200, 100, 50),
        ];

        for (r, g, b) in test_colors {
            let (l, a, bv) = rgb_to_lab(r, g, b);
            let (r2, g2, b2) = lab_to_rgb(l, a, bv);
            assert!(
                (r as i16 - r2 as i16).abs() <= 1
                    && (g as i16 - g2 as i16).abs() <= 1
                    && (b as i16 - b2 as i16).abs() <= 1,
                "RGB ({},{},{}) -> LAB ({:.1},{:.1},{:.1}) -> RGB ({},{},{})",
                r, g, b, l, a, bv, r2, g2, b2,
            );
        }
    }

    #[test]
    fn test_color_transfer_identity() {
        let img = Array3::from_shape_fn((10, 10, 3), |(y, x, c)| {
            ((y * 25 + x * 25 + c * 80) % 256) as u8
        });
        let result = color_transfer_lab(&img, &img);
        for y in 0..10 {
            for x in 0..10 {
                for c in 0..3 {
                    assert!(
                        (img[[y, x, c]] as i16 - result[[y, x, c]] as i16).abs() <= 2,
                        "Pixel ({},{},{}) differs: {} vs {}",
                        y, x, c, img[[y, x, c]], result[[y, x, c]],
                    );
                }
            }
        }
    }
}
