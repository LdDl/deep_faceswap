//! Computer vision utility functions

use crate::types::{BBox, DetectedFace};
use ndarray::Array2;

/// Compute Intersection over Union (IoU) between two bounding boxes
///
/// # Arguments
/// * `box1` - First bounding box
/// * `box2` - Second bounding box
///
/// # Returns
/// IoU value in range [0.0, 1.0], where 0.0 means no overlap and 1.0 means perfect overlap
pub fn compute_iou(box1: &BBox, box2: &BBox) -> f32 {
    let x1 = box1.x1.max(box2.x1);
    let y1 = box1.y1.max(box2.y1);
    let x2 = box1.x2.min(box2.x2);
    let y2 = box1.y2.min(box2.y2);

    if x2 <= x1 || y2 <= y1 {
        return 0.0;
    }

    let intersection = (x2 - x1) * (y2 - y1);
    let area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    let area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    let union = area1 + area2 - intersection;

    if union <= 0.0 {
        return 0.0;
    }

    intersection / union
}

/// Apply Non-Maximum Suppression (NMS) to filter overlapping detections
///
/// NMS keeps only the highest-confidence detection among overlapping ones.
/// Detections are sorted internally by confidence score before processing.
/// Strictly taken from here: https://github.com/LdDl/object-detection-opencv-rust/blob/master/src/postprocess.rs#L42
///
/// # Arguments
/// * `detections` - List of detected faces
/// * `iou_threshold` - IoU threshold for suppression (typically 0.3-0.5)
///
/// # Returns
/// Filtered list of detections with overlaps removed
pub fn apply_nms(detections: &[DetectedFace], iou_threshold: f32) -> Vec<DetectedFace> {
    if detections.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<_> = detections.iter().enumerate().collect();
    sorted.sort_by(|a, b| {
        b.1.det_score
            .partial_cmp(&a.1.det_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep = Vec::new();
    let mut suppressed = vec![false; detections.len()];

    for (orig_idx, detection) in sorted.iter() {
        if suppressed[*orig_idx] {
            continue;
        }

        keep.push((*detection).clone());

        for (other_orig_idx, other) in sorted.iter() {
            if suppressed[*other_orig_idx] || orig_idx == other_orig_idx {
                continue;
            }

            let iou = compute_iou(&detection.bbox, &other.bbox);
            if iou > iou_threshold {
                suppressed[*other_orig_idx] = true;
            }
        }
    }

    keep
}

/// Apply erosion morphological operation to a 2D mask
///
/// Erosion shrinks white regions by removing pixels at their boundaries.
/// A pixel remains white (255) only if ALL pixels in its kernel neighborhood are white.
/// This operation creates an inward offset from mask edges, useful for avoiding hard boundaries.
///
/// # Arguments
/// * `mask` - Input 2D mask (u8 values, typically 0 or 255)
/// * `kernel_size` - Size of square erosion kernel (must be odd, typically 3-21)
///
/// # Returns
/// Eroded mask with same dimensions as input
///
/// # Example
/// ```
/// use ndarray::Array2;
/// use deep_faceswap_core::utils::cv::erode_mask;
///
/// let mut mask = Array2::<u8>::zeros((10, 10));
/// // Create a 6x6 white square in the center
/// for y in 2..8 {
///     for x in 2..8 {
///         mask[[y, x]] = 255;
///     }
/// }
///
/// let eroded = erode_mask(&mask, 3);
///
/// // Center remains white
/// assert_eq!(eroded[[5, 5]], 255);
/// // Edges are removed (turned black)
/// assert_eq!(eroded[[2, 2]], 0);
/// ```
pub fn erode_mask(mask: &Array2<u8>, kernel_size: usize) -> Array2<u8> {
    assert!(kernel_size >= 3 && kernel_size % 2 == 1, "Kernel size must be odd and >= 3");

    let (h, w) = (mask.nrows(), mask.ncols());
    let mut output = Array2::<u8>::zeros((h, w));
    let half = (kernel_size / 2) as isize;

    for y in 0..h {
        for x in 0..w {
            let mut all_white = true;

            // Check all pixels in kernel neighborhood
            for ky in -(half)..=(half) {
                for kx in -(half)..=(half) {
                    let ny = (y as isize + ky).max(0).min((h - 1) as isize) as usize;
                    let nx = (x as isize + kx).max(0).min((w - 1) as isize) as usize;

                    if mask[[ny, nx]] < 255 {
                        all_white = false;
                        break;
                    }
                }
                if !all_white {
                    break;
                }
            }

            output[[y, x]] = if all_white { 255 } else { 0 };
        }
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-6;

    fn make_bbox(x1: f32, y1: f32, x2: f32, y2: f32) -> BBox {
        BBox {
            x1,
            y1,
            x2,
            y2,
            score: 0.9,
        }
    }

    fn make_face(x1: f32, y1: f32, x2: f32, y2: f32, score: f32) -> DetectedFace {
        DetectedFace {
            bbox: BBox {
                x1,
                y1,
                x2,
                y2,
                score,
            },
            landmarks: [[0.0, 0.0]; 5],
            det_score: score,
        }
    }

    #[test]
    fn test_compute_iou_identical_boxes() {
        let box1 = make_bbox(0.0, 0.0, 10.0, 10.0);
        let box2 = make_bbox(0.0, 0.0, 10.0, 10.0);
        let iou = compute_iou(&box1, &box2);
        assert!(
            (iou - 1.0).abs() < EPS,
            "Identical boxes should have IoU = 1.0"
        );
    }

    #[test]
    fn test_compute_iou_no_overlap() {
        let box1 = make_bbox(0.0, 0.0, 10.0, 10.0);
        let box2 = make_bbox(20.0, 20.0, 30.0, 30.0);
        let iou = compute_iou(&box1, &box2);
        assert_eq!(iou, 0.0, "Non-overlapping boxes should have IoU = 0.0");
    }

    #[test]
    fn test_compute_iou_partial_overlap() {
        let box1 = make_bbox(0.0, 0.0, 10.0, 10.0);
        let box2 = make_bbox(5.0, 5.0, 15.0, 15.0);
        let iou = compute_iou(&box1, &box2);

        let intersection = 5.0 * 5.0;
        let area1 = 10.0 * 10.0;
        let area2 = 10.0 * 10.0;
        let union = area1 + area2 - intersection;
        let expected_iou = intersection / union;

        assert!(
            (iou - expected_iou).abs() < EPS,
            "Partial overlap IoU mismatch"
        );
    }

    #[test]
    fn test_compute_iou_contained() {
        let box1 = make_bbox(0.0, 0.0, 20.0, 20.0);
        let box2 = make_bbox(5.0, 5.0, 15.0, 15.0);
        let iou = compute_iou(&box1, &box2);

        let intersection = 10.0 * 10.0;
        let area1 = 20.0 * 20.0;
        let area2 = 10.0 * 10.0;
        let union = area1 + area2 - intersection;
        let expected_iou = intersection / union;

        assert!(
            (iou - expected_iou).abs() < EPS,
            "Contained box IoU mismatch"
        );
    }

    #[test]
    fn test_compute_iou_edge_touch() {
        let box1 = make_bbox(0.0, 0.0, 10.0, 10.0);
        let box2 = make_bbox(10.0, 0.0, 20.0, 10.0);
        let iou = compute_iou(&box1, &box2);
        assert_eq!(iou, 0.0, "Edge-touching boxes should have IoU = 0.0");
    }

    #[test]
    fn test_apply_nms_empty() {
        let detections: Vec<DetectedFace> = vec![];
        let result = apply_nms(&detections, 0.5);
        assert_eq!(result.len(), 0, "Empty input should return empty output");
    }

    #[test]
    fn test_apply_nms_single_detection() {
        let detections = vec![make_face(0.0, 0.0, 10.0, 10.0, 0.9)];
        let result = apply_nms(&detections, 0.5);
        assert_eq!(result.len(), 1, "Single detection should pass through");
    }

    #[test]
    fn test_apply_nms_no_overlap() {
        let detections = vec![
            make_face(0.0, 0.0, 10.0, 10.0, 0.9),
            make_face(20.0, 20.0, 30.0, 30.0, 0.8),
            make_face(40.0, 40.0, 50.0, 50.0, 0.7),
        ];
        let result = apply_nms(&detections, 0.5);
        assert_eq!(
            result.len(),
            3,
            "Non-overlapping detections should all be kept"
        );
    }

    #[test]
    fn test_apply_nms_high_overlap() {
        let detections = vec![
            make_face(0.0, 0.0, 10.0, 10.0, 0.9),
            make_face(0.5, 0.5, 10.5, 10.5, 0.8),
            make_face(1.0, 1.0, 11.0, 11.0, 0.7),
        ];
        let result = apply_nms(&detections, 0.5);
        assert_eq!(
            result.len(),
            1,
            "Highly overlapping detections should be suppressed"
        );
        assert!(
            (result[0].det_score - 0.9).abs() < EPS,
            "Highest score detection should be kept"
        );
    }

    #[test]
    fn test_apply_nms_threshold() {
        let detections = vec![
            make_face(0.0, 0.0, 10.0, 10.0, 0.9),
            make_face(5.0, 5.0, 15.0, 15.0, 0.8),
        ];

        let result_low_thresh = apply_nms(&detections, 0.1);
        assert_eq!(
            result_low_thresh.len(),
            1,
            "Low threshold should suppress overlap"
        );

        let result_high_thresh = apply_nms(&detections, 0.9);
        assert_eq!(
            result_high_thresh.len(),
            2,
            "High threshold should allow overlap"
        );
    }

    #[test]
    fn test_apply_nms_preserves_order() {
        let detections = vec![
            make_face(0.0, 0.0, 10.0, 10.0, 0.9),
            make_face(20.0, 20.0, 30.0, 30.0, 0.7),
            make_face(40.0, 40.0, 50.0, 50.0, 0.8),
        ];
        let result = apply_nms(&detections, 0.5);

        assert_eq!(result.len(), 3);
        assert!((result[0].det_score - 0.9).abs() < EPS);
        assert!((result[1].det_score - 0.8).abs() < EPS);
        assert!((result[2].det_score - 0.7).abs() < EPS);
    }

    #[test]
    fn test_erode_mask_square() {
        let mut mask = Array2::<u8>::zeros((10, 10));
        for y in 2..8 {
            for x in 2..8 {
                mask[[y, x]] = 255;
            }
        }

        let eroded = erode_mask(&mask, 3);

        assert_eq!(eroded[[4, 4]], 255, "Center should remain white");
        assert_eq!(eroded[[2, 2]], 0, "Corner should be eroded");
        assert_eq!(eroded[[2, 5]], 0, "Edge should be eroded");
    }

    #[test]
    fn test_erode_mask_small_object_disappears() {
        let mut mask = Array2::<u8>::zeros((10, 10));
        mask[[5, 5]] = 255;
        mask[[5, 6]] = 255;

        let eroded = erode_mask(&mask, 3);

        assert_eq!(eroded[[5, 5]], 0, "Small object should disappear");
        assert_eq!(eroded[[5, 6]], 0, "Small object should disappear");
    }

    #[test]
    fn test_erode_mask_full_image() {
        let mask = Array2::<u8>::from_elem((10, 10), 255);
        let eroded = erode_mask(&mask, 3);

        assert_eq!(eroded[[5, 5]], 255, "Center should remain white");
        assert_eq!(eroded[[0, 0]], 255, "Corner remains white (border replicate)");
        assert_eq!(eroded[[0, 5]], 255, "Edge remains white (border replicate)");
    }

    #[test]
    fn test_erode_mask_larger_kernel() {
        let mut mask = Array2::<u8>::zeros((20, 20));
        for y in 5..15 {
            for x in 5..15 {
                mask[[y, x]] = 255;
            }
        }

        let eroded = erode_mask(&mask, 5);

        assert_eq!(eroded[[10, 10]], 255, "Center should remain white");
        assert_eq!(eroded[[5, 5]], 0, "Corner should be heavily eroded");
    }

    #[test]
    #[should_panic(expected = "Kernel size must be odd")]
    fn test_erode_mask_even_kernel_panics() {
        let mask = Array2::<u8>::zeros((10, 10));
        erode_mask(&mask, 4);
    }

    #[test]
    #[should_panic(expected = "Kernel size must be odd")]
    fn test_erode_mask_too_small_kernel_panics() {
        let mask = Array2::<u8>::zeros((10, 10));
        erode_mask(&mask, 1);
    }
}
