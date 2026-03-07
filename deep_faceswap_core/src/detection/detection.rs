//! Face detection using SCRFD from buffalo_l package
//!
//! Model: det_10g.onnx (SCRFD detector with 3 scales)

use crate::types::{BBox, DetectedFace, FaceSwapError, Result};
use crate::utils::cv::apply_nms;
use crate::verbose::{get_verbose_level, VerboseLevel, EVENT_DETECT_FACES, EVENT_LOAD_MODEL};
use crate::{log_additional, log_main};
use ndarray::{Array3, Array4, ArrayD};
use ort::{
    inputs,
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use std::path::Path;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;

const INPUT_WIDTH: u32 = 640;
const INPUT_HEIGHT: u32 = 640;

/// Face detector using SCRFD (Sample and Computation Redistribution for Face Detection)
///
/// This detector uses the buffalo_l model package from InsightFace, specifically the
/// det_10g.onnx model which is a SCRFD detector operating at 3 different scales.
/// https://github.com/deepinsight/insightface/blob/master/detection/scrfd/README.md
///
/// # Model details
/// - Input size: 640x640 pixels
/// - Feature pyramid network with strides: 8, 16, 32
/// - 2 anchors per grid position
/// - Outputs: bounding boxes, confidence scores, and 5 facial landmarks per detection
///
/// # Example
/// ```ignore
/// use deep_faceswap_core::detection::FaceDetector;
/// use deep_faceswap_core::alignment::load_image;
///
/// let mut detector = FaceDetector::new("models/buffalo_l/det_10g.onnx")?;
/// let image = load_image("face.jpg")?;
/// let image_array = /* convert to Array3<u8> */;
/// let faces = detector.detect(&image_array, 0.5, 0.4)?;
/// ```
pub struct FaceDetector {
    session: Session,
    input_width: u32,
    input_height: u32,
    feat_stride_fpn: Vec<usize>,
    num_anchors: usize,
}

impl FaceDetector {
    /// Load face detector from ONNX model
    pub fn new(model_path: &str) -> Result<Self> {
        if !Path::new(model_path).exists() {
            return Err(FaceSwapError::ModelNotFound(model_path.to_string()));
        }

        log_main!(EVENT_LOAD_MODEL, "Loading face detector", path = model_path);

        let ort_log_level = match get_verbose_level() {
            VerboseLevel::None => LogLevel::Fatal,
            VerboseLevel::Main => LogLevel::Fatal,
            VerboseLevel::Additional => LogLevel::Warning,
            VerboseLevel::All => LogLevel::Verbose,
        };

        #[cfg(feature = "cuda")]
        let session = Session::builder()
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_log_level(ort_log_level)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_execution_providers([CUDAExecutionProvider::default().build()])
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

        #[cfg(not(feature = "cuda"))]
        let session = Session::builder()
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_log_level(ort_log_level)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
            .commit_from_file(model_path)
            .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

        Ok(Self {
            session,
            input_width: INPUT_WIDTH,
            input_height: INPUT_HEIGHT,
            feat_stride_fpn: vec![8, 16, 32],
            num_anchors: 2,
        })
    }

    /// Detect faces in image
    ///
    /// # Arguments
    /// * `image` - Input image as HWC array (H, W, 3) with values in [0, 255]
    /// * `conf_threshold` - Confidence threshold [0.0; 1.0]
    /// * `iou_threshold` - IoU threshold for NMS [0.0; 1.0]
    ///
    /// # Returns
    /// Vector of detected faces sorted by score (descending)
    pub fn detect(
        &mut self,
        image: &Array3<u8>,
        conf_threshold: f32,
        iou_threshold: f32,
    ) -> Result<Vec<DetectedFace>> {
        log_additional!(EVENT_DETECT_FACES, "Detecting faces");

        let (input_tensor, det_scale) =
            preprocess_image(image, self.input_width, self.input_height)?;

        let (scores, bboxes, kpss) = {
            let outputs = self
                .session
                .run(inputs!["input.1" => TensorRef::from_array_view(&input_tensor)?])
                .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?;

            // Extract all outputs (3 scales × 3 types)
            let scores = vec![
                outputs["448"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
                outputs["471"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
                outputs["494"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
            ];

            let bboxes = vec![
                outputs["451"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
                outputs["474"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
                outputs["497"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
            ];

            let kpss = vec![
                outputs["454"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
                outputs["477"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
                outputs["500"]
                    .try_extract_array::<f32>()
                    .map_err(|e| FaceSwapError::Ort(format!("{}", e)))?
                    .into_owned(),
            ];

            (scores, bboxes, kpss)
        };

        let detections =
            self.parse_scrfd_outputs(&scores, &bboxes, &kpss, conf_threshold, det_scale)?;
        let filtered = apply_nms(&detections, iou_threshold);

        Ok(filtered)
    }

    fn parse_scrfd_outputs(
        &self,
        scores_list: &[ArrayD<f32>],
        bboxes_list: &[ArrayD<f32>],
        kpss_list: &[ArrayD<f32>],
        conf_threshold: f32,
        det_scale: f32,
    ) -> Result<Vec<DetectedFace>> {
        let mut detections = Vec::new();

        for (idx, stride) in self.feat_stride_fpn.iter().enumerate() {
            let scores = &scores_list[idx];
            let bboxes = &bboxes_list[idx];
            let kpss = &kpss_list[idx];

            let num_proposals = scores.shape()[0];
            let width = (self.input_width as usize) / stride;

            for i in 0..num_proposals {
                let score = scores[[i, 0]];
                if score < conf_threshold {
                    continue;
                }

                // Account for num_anchors: each grid position has multiple anchors
                let anchor_idx = i / self.num_anchors;
                let grid_y = anchor_idx / width;
                let grid_x = anchor_idx % width;

                let center_x = (grid_x as f32) * (*stride as f32);
                let center_y = (grid_y as f32) * (*stride as f32);

                let distance0 = bboxes[[i, 0]] * (*stride as f32);
                let distance1 = bboxes[[i, 1]] * (*stride as f32);
                let distance2 = bboxes[[i, 2]] * (*stride as f32);
                let distance3 = bboxes[[i, 3]] * (*stride as f32);

                let x1 = (center_x - distance0) / det_scale;
                let y1 = (center_y - distance1) / det_scale;
                let x2 = (center_x + distance2) / det_scale;
                let y2 = (center_y + distance3) / det_scale;

                let mut landmarks = [[0.0f32; 2]; 5];
                for j in 0..5 {
                    landmarks[j][0] = (center_x + kpss[[i, j * 2]] * (*stride as f32)) / det_scale;
                    landmarks[j][1] =
                        (center_y + kpss[[i, j * 2 + 1]] * (*stride as f32)) / det_scale;
                }

                detections.push(DetectedFace {
                    bbox: BBox {
                        x1,
                        y1,
                        x2,
                        y2,
                        score,
                    },
                    landmarks,
                    det_score: score,
                });
            }
        }

        detections.sort_by(|a, b| b.det_score.partial_cmp(&a.det_score).unwrap());

        Ok(detections)
    }
}

fn preprocess_image(
    image: &Array3<u8>,
    input_width: u32,
    input_height: u32,
) -> Result<(Array4<f32>, f32)> {
    let (h, w, c) = (image.shape()[0], image.shape()[1], image.shape()[2]);

    if c != 3 {
        return Err(FaceSwapError::InvalidInput(format!(
            "Expected 3 channels, got {}",
            c
        )));
    }

    // Calculate scale to maintain aspect ratio
    let im_ratio = h as f32 / w as f32;
    let model_ratio = input_height as f32 / input_width as f32;

    let (new_width, new_height) = if im_ratio > model_ratio {
        let new_height = input_height as usize;
        let new_width = (new_height as f32 / im_ratio) as usize;
        (new_width, new_height)
    } else {
        let new_width = input_width as usize;
        let new_height = (new_width as f32 * im_ratio) as usize;
        (new_width, new_height)
    };

    let det_scale = new_height as f32 / h as f32;

    // Create zero-padded input
    let mut input = Array4::<f32>::zeros((1, 3, input_height as usize, input_width as usize));

    // Resize and copy only the valid region
    let scale = det_scale;
    for y in 0..new_height {
        for x in 0..new_width {
            let src_x = ((x as f32 / scale) as usize).min(w - 1);
            let src_y = ((y as f32 / scale) as usize).min(h - 1);

            // Normalize to [-1, 1]
            input[[0, 0, y, x]] = (image[[src_y, src_x, 0]] as f32 - 127.5) / 128.0;
            input[[0, 1, y, x]] = (image[[src_y, src_x, 1]] as f32 - 127.5) / 128.0;
            input[[0, 2, y, x]] = (image[[src_y, src_x, 2]] as f32 - 127.5) / 128.0;
        }
    }

    Ok((input, det_scale))
}
