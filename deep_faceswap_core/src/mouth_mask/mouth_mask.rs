//! Mouth mask creation and application
//!
//! When swapping faces, the source's mouth expression replaces the target's.
//! If the target has an open mouth but the source has a closed mouth, the result
//! looks unnatural. The mouth mask preserves the target's mouth region by blending
//! it back into the swapped result.
//!
//! # Pipeline
//! 1. Before swap: extract mouth region from original target image using 106 landmarks
//! 2. After swap: blend the original mouth back into the swapped result
//!
//! # Landmark indices used (from 106-point set)
//! Lower lip + chin polygon: [65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2]

use crate::types::Result;
use crate::utils::blur::gaussian_blur_2d;
use crate::utils::color::color_transfer_lab;
use ndarray::{s, Array2, Array3};

// Landmark indices defining the mouth polygon (from 106-point set)
const LOWER_LIP_ORDER: [usize; 21] = [
    65, 66, 62, 70, 69, 18, 19, 20, 21, 22, 23, 24, 0, 8, 7, 6, 5, 4, 3, 2, 65,
];

// Indices within LOWER_LIP_ORDER array (not landmark indices!) for top lip
// It maps to landmarks [2, 65, 66, 62, 70, 69, 18]
const TOP_LIP_INDICES: [usize; 7] = [20, 0, 1, 2, 3, 4, 5];

// Indices within LOWER_LIP_ORDER array (not landmark indices!) for chin
// It maps to landmarks [24, 0, 8, 7, 6, 5]
const CHIN_INDICES: [usize; 6] = [11, 12, 13, 14, 15, 16];

// Expansion factor for the polygon outward from center
const EXPANSION_FACTOR: f32 = 1.1;

// Top lip extension scale
const TOP_LIP_EXTENSION: f32 = 0.5;

// Chin extension scale
const CHIN_EXTENSION: f32 = 0.4;

// Padding ratio around mouth bounding box
const PADDING_RATIO: f32 = 0.1;

// Default Gaussian blur kernel size for mask creation
const MASK_BLUR_KERNEL: usize = 15;

// Default feather ratio for mask application
const MASK_FEATHER_RATIO: usize = 12;

// Maximum feather amount in pixels
const MAX_FEATHER: usize = 30;

pub struct MouthMaskData {
    /// Blurred mask for the mouth region (ROI-sized, float values)
    pub mask: Array2<f32>,
    /// Mouth cutout from the original target image (HWC u8)
    pub mouth_cutout: Array3<u8>,
    /// Bounding box of the mouth region [min_x, min_y, max_x, max_y]
    pub mouth_box: [usize; 4],
    /// Expanded polygon points in original image coordinates
    pub polygon: Vec<[f32; 2]>,
}

/// Create mouth mask from 106-point landmarks and the original target image.
///
/// Returns the mask data needed to later apply the mouth blending after face swap.
/// The mask is ROI-sized (only covers the mouth region, not the full frame).
pub fn create_mouth_mask(image: &Array3<u8>, landmarks: &[[f32; 2]]) -> Result<MouthMaskData> {
    let (frame_h, frame_w, _) = image.dim();

    // Extract mouth polygon landmarks
    let mut mouth_points: Vec<[f32; 2]> =
        LOWER_LIP_ORDER.iter().map(|&idx| landmarks[idx]).collect();

    // Calculate polygon center
    let center_x: f32 = mouth_points.iter().map(|p| p[0]).sum::<f32>() / mouth_points.len() as f32;
    let center_y: f32 = mouth_points.iter().map(|p| p[1]).sum::<f32>() / mouth_points.len() as f32;

    // Expand landmarks outward from center
    for pt in mouth_points.iter_mut() {
        pt[0] = (pt[0] - center_x) * EXPANSION_FACTOR + center_x;
        pt[1] = (pt[1] - center_y) * EXPANSION_FACTOR + center_y;
    }

    // Extend top lip points upward (away from center)
    for &idx in &TOP_LIP_INDICES {
        let dx = mouth_points[idx][0] - center_x;
        let dy = mouth_points[idx][1] - center_y;
        let dist = (dx * dx + dy * dy).sqrt();
        if dist > 1e-6 {
            let nx = dx / dist;
            let ny = dy / dist;
            mouth_points[idx][0] += nx * TOP_LIP_EXTENSION;
            mouth_points[idx][1] += ny * TOP_LIP_EXTENSION;
        }
    }

    // Extend chin points downward
    for &idx in &CHIN_INDICES {
        let dy = mouth_points[idx][1] - center_y;
        mouth_points[idx][1] += dy * CHIN_EXTENSION;
    }

    // Calculate bounding box with padding
    let mut min_x = f32::MAX;
    let mut min_y = f32::MAX;
    let mut max_x = f32::MIN;
    let mut max_y = f32::MIN;
    for pt in &mouth_points {
        min_x = min_x.min(pt[0]);
        min_y = min_y.min(pt[1]);
        max_x = max_x.max(pt[0]);
        max_y = max_y.max(pt[1]);
    }

    let box_w = max_x - min_x;
    let box_h = max_y - min_y;
    let pad_x = box_w * PADDING_RATIO;
    let pad_y = box_h * PADDING_RATIO;

    let min_x = (min_x - pad_x).max(0.0) as usize;
    let min_y = (min_y - pad_y).max(0.0) as usize;
    let max_x = ((max_x + pad_x).min(frame_w as f32 - 1.0) as usize + 1).min(frame_w);
    let max_y = ((max_y + pad_y).min(frame_h as f32 - 1.0) as usize + 1).min(frame_h);

    let roi_w = max_x - min_x;
    let roi_h = max_y - min_y;

    // Create mask ROI with filled polygon
    let mut mask_roi = Array2::<f32>::zeros((roi_h, roi_w));
    fill_polygon_on_mask(&mut mask_roi, &mouth_points, min_x as f32, min_y as f32);

    // Apply Gaussian blur to soften edges
    let mask_roi = gaussian_blur_2d(&mask_roi, MASK_BLUR_KERNEL);

    // Extract mouth cutout from original image
    let mouth_cutout = image.slice(s![min_y..max_y, min_x..max_x, ..]).to_owned();

    // Return ROI-sized mask directly (no full-frame allocation)
    Ok(MouthMaskData {
        mask: mask_roi,
        mouth_cutout,
        mouth_box: [min_x, min_y, max_x, max_y],
        polygon: mouth_points,
    })
}

/// Apply the mouth mask: blend original mouth region back into the swapped frame.
///
/// So, the pipeline is:
/// 1. Color-corrects the original mouth to match swapped face colors
/// 2. Creates a feathered polygon mask
/// 3. Alpha-blends the original mouth back into the result
pub fn apply_mouth_mask(frame: &mut Array3<u8>, mouth_data: &MouthMaskData) {
    let [min_x, min_y, max_x, max_y] = mouth_data.mouth_box;
    let roi_w = max_x - min_x;
    let roi_h = max_y - min_y;

    if roi_w == 0 || roi_h == 0 {
        return;
    }

    // Extract current ROI from the swapped frame
    let roi = frame.slice(s![min_y..max_y, min_x..max_x, ..]).to_owned();

    // Color-correct the original mouth cutout to match swapped face colors
    let color_corrected = color_transfer_lab(&mouth_data.mouth_cutout, &roi);

    // Create feathered polygon mask for blending
    let feather_base = roi_w.min(roi_h);
    let feather_amount = 1.max((feather_base / MASK_FEATHER_RATIO).min(MAX_FEATHER));
    let kernel_size = 2 * feather_amount + 1;
    // Make sure kernel is odd and at least 3
    let kernel_size = if kernel_size < 3 { 3 } else { kernel_size | 1 };

    let mut polygon_mask = Array2::<f32>::zeros((roi_h, roi_w));
    fill_polygon_on_mask(
        &mut polygon_mask,
        &mouth_data.polygon,
        min_x as f32,
        min_y as f32,
    );

    let feathered_mask = gaussian_blur_2d(&polygon_mask, kernel_size);

    // Normalize feathered mask to [0, 1]
    let max_val = feathered_mask.iter().cloned().fold(0.0f32, f32::max);
    let feathered_mask = if max_val > 1e-6 {
        feathered_mask.mapv(|v| v / max_val)
    } else {
        feathered_mask
    };

    // Combine with the face mask (already ROI-sized from create_mouth_mask)
    // Take minimum to ensure mouth blending stays within face area
    let combined_mask = Array2::from_shape_fn((roi_h, roi_w), |(y, x)| {
        feathered_mask[[y, x]].min(mouth_data.mask[[y, x]])
    });

    // Alpha blend: result = mouth * mask + swapped * (1 - mask)
    for y in 0..roi_h {
        for x in 0..roi_w {
            let alpha = combined_mask[[y, x]];
            if alpha > 0.0 {
                for c in 0..3 {
                    let mouth_val = color_corrected[[y, x, c]] as f32;
                    let swap_val = frame[[min_y + y, min_x + x, c]] as f32;
                    let blended = mouth_val * alpha + swap_val * (1.0 - alpha);
                    frame[[min_y + y, min_x + x, c]] = blended.clamp(0.0, 255.0) as u8;
                }
            }
        }
    }
}

// Fill a polygon on a 2D mask using scanline rasterization.
// Polygon points are in original image coordinates,
// offset_x/offset_y are subtracted to convert to ROI-local coordinates.
// @todo: may be make public and more optimizations?
fn fill_polygon_on_mask(mask: &mut Array2<f32>, points: &[[f32; 2]], offset_x: f32, offset_y: f32) {
    let (h, w) = (mask.nrows(), mask.ncols());
    let n = points.len();
    if n < 3 {
        return;
    }

    // Convert to ROI-local coordinates
    let local: Vec<(f32, f32)> = points
        .iter()
        .map(|p| (p[0] - offset_x, p[1] - offset_y))
        .collect();

    // Find Y range
    let min_y = local.iter().map(|p| p.1).fold(f32::MAX, f32::min).max(0.0) as usize;
    let max_y = local
        .iter()
        .map(|p| p.1)
        .fold(f32::MIN, f32::max)
        .min((h - 1) as f32) as usize;

    // Scanline fill
    for y in min_y..=max_y {
        let yf = y as f32 + 0.5;
        let mut intersections = Vec::new();

        for i in 0..n {
            let j = (i + 1) % n;
            let (x0, y0) = local[i];
            let (x1, y1) = local[j];

            if (y0 <= yf && y1 > yf) || (y1 <= yf && y0 > yf) {
                let t = (yf - y0) / (y1 - y0);
                let x_intersect = x0 + t * (x1 - x0);
                intersections.push(x_intersect);
            }
        }

        intersections.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for pair in intersections.chunks(2) {
            if pair.len() == 2 {
                let x_start = (pair[0].max(0.0) as usize).min(w);
                let x_end = ((pair[1] + 1.0).max(0.0) as usize).min(w);
                for x in x_start..x_end {
                    mask[[y, x]] = 1.0;
                }
            }
        }
    }
}
