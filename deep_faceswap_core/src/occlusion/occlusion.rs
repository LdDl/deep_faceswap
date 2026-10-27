//! Foreground preservation using a visible-face mask from an XSeg ONNX model.

use crate::alignment::{align_face, face_region};
use crate::log_main;
use crate::types::{DetectedFace, FaceSwapError, Result};
use crate::utils::{blur::gaussian_blur_2d, cv::erode_mask_optimized};
use crate::verbose::{get_verbose_level, VerboseLevel, EVENT_LOAD_MODEL};
use ndarray::{s, Array2, Array3, Array4};
use ort::{
    inputs,
    logging::LogLevel,
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[cfg(feature = "cuda")]
use ort::execution_providers::{cuda::ConvAlgorithmSearch, CUDAExecutionProvider};

const MASK_SIZE: usize = 256;

/// Mask adjustments measured in pixels of the 256x256 aligned face.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(default)]
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
pub struct OcclusionOptions {
    /// Expand protected objects by this radius, from 0 to 16 pixels.
    pub margin: usize,
    /// Soften the face side of protected edges, from 0 to 16 pixels.
    pub blur: usize,
}

impl Default for OcclusionOptions {
    fn default() -> Self {
        Self { margin: 2, blur: 3 }
    }
}

impl OcclusionOptions {
    pub fn validate(&self) -> Result<()> {
        if self.margin > 16 || self.blur > 16 {
            return Err(FaceSwapError::InvalidInput(
                "Foreground margin and edge softness must be between 0 and 16".to_string(),
            ));
        }
        Ok(())
    }
}

/// XSeg model accepting NHWC BGR floats in [0, 1] and returning a visible-face mask.
pub struct FaceOccluder {
    session: Session,
}

impl FaceOccluder {
    pub fn new(model_path: &str) -> Result<Self> {
        if !Path::new(model_path).is_file() {
            return Err(FaceSwapError::ModelNotFound(model_path.to_string()));
        }
        log_main!(EVENT_LOAD_MODEL, "Loading face occluder", path = model_path);
        let log_level = match get_verbose_level() {
            VerboseLevel::None | VerboseLevel::Main => LogLevel::Fatal,
            VerboseLevel::Additional => LogLevel::Warning,
            VerboseLevel::All => LogLevel::Verbose,
        };
        let mut builder = Session::builder()?
            .with_log_level(log_level)
            .map_err(|e| FaceSwapError::Ort(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| FaceSwapError::Ort(e.to_string()))?;
        #[cfg(feature = "cuda")]
        {
            builder = builder
                .with_execution_providers([CUDAExecutionProvider::default()
                    .with_tf32(true)
                    .with_conv_algorithm_search(ConvAlgorithmSearch::Heuristic)
                    .build()])
                .map_err(|e| FaceSwapError::Ort(e.to_string()))?;
        }
        let session = builder.commit_from_file(model_path)?;
        Ok(Self { session })
    }

    /// Infer a 256x256 mask from an aligned NCHW RGB face in [0, 1]. One means visible face.
    pub fn visible_mask(&mut self, aligned_face: &Array4<f32>) -> Result<Array2<f32>> {
        let input = prepare_input(aligned_face)?;
        let outputs = self
            .session
            .run(inputs!["input" => TensorRef::from_array_view(&input)?])?;
        let mask = outputs[0].try_extract_array::<f32>()?;
        if mask.shape() != [1, MASK_SIZE, MASK_SIZE, 1] && mask.shape() != [1, MASK_SIZE, MASK_SIZE]
        {
            return Err(FaceSwapError::ProcessingError(format!(
                "Expected XSeg output shape [1, 256, 256, 1] or [1, 256, 256], got {:?}",
                mask.shape()
            )));
        }
        let mask = Array2::from_shape_vec((MASK_SIZE, MASK_SIZE), mask.iter().copied().collect())
            .map_err(|e| FaceSwapError::ProcessingError(e.to_string()))?;
        if mask.iter().any(|value| !value.is_finite()) {
            return Err(FaceSwapError::ProcessingError(
                "XSeg returned a non-finite mask".to_string(),
            ));
        }
        Ok(mask.mapv(|value| value.clamp(0.0, 1.0)))
    }
}

fn prepare_input(aligned_face: &Array4<f32>) -> Result<Array4<f32>> {
    if aligned_face.shape() != [1, 3, MASK_SIZE, MASK_SIZE] {
        return Err(FaceSwapError::InvalidInput(
            "XSeg requires a 256x256 aligned RGB face".to_string(),
        ));
    }
    Ok(Array4::from_shape_fn(
        (1, MASK_SIZE, MASK_SIZE, 3),
        |(_, y, x, c)| aligned_face[[0, 2 - c, y, x]],
    ))
}

/// A model and per-operation settings shared by all faces and frames in one swap.
pub struct ForegroundProtection<'a> {
    pub occluder: &'a mut FaceOccluder,
    pub options: OcclusionOptions,
}

impl ForegroundProtection<'_> {
    pub(crate) fn capture(
        &mut self,
        image: &Array3<u8>,
        face: &DetectedFace,
    ) -> Result<ProtectedRegion> {
        self.options.validate()?;
        let aligned = align_face(image, face, MASK_SIZE as u32)?;
        let visible = self.occluder.visible_mask(&aligned.aligned_image)?;
        let visible = adjust_mask(&visible, self.options);
        ProtectedRegion::capture(image, &visible, &aligned.transform)
    }
}

fn adjust_mask(visible: &Array2<f32>, options: OcclusionOptions) -> Array2<f32> {
    let visible = visible.mapv(|value| (value * 255.0).round() as u8);
    let visible = if options.margin > 0 {
        erode_mask_optimized(&visible, options.margin * 2 + 1)
    } else {
        visible
    };
    let visible = visible.mapv(|value| ((value as f32 / 255.0 - 0.5) * 2.0).clamp(0.0, 1.0));
    if options.blur == 0 {
        return visible;
    }
    let blurred = gaussian_blur_2d(&visible, options.blur * 2 + 1);
    // Feather into the face without allowing blur to uncover a protected pixel.
    Array2::from_shape_fn(visible.dim(), |index| visible[index].min(blurred[index]))
}

/// Original pixels and a frame-space visible-face mask covering the paste-back ROI.
pub(crate) struct ProtectedRegion {
    original: Array3<u8>,
    visible: Array2<f32>,
    x: usize,
    y: usize,
}

impl ProtectedRegion {
    fn capture(image: &Array3<u8>, mask: &Array2<f32>, transform: &Array2<f32>) -> Result<Self> {
        let (height, width, _) = image.dim();
        let region = face_region(transform, MASK_SIZE as u32, width, height)?;
        // Cover rounding differences between the 128, 256 and 512 pixel alignment templates.
        let x = region.x.saturating_sub(2);
        let y = region.y.saturating_sub(2);
        let end_x = (region.x + region.width + 2).min(width);
        let end_y = (region.y + region.height + 2).min(height);
        let original = image.slice(s![y..end_y, x..end_x, ..]).to_owned();
        let visible = Array2::from_shape_fn((end_y - y, end_x - x), |(row, col)| {
            let frame_x = (x + col) as f32;
            let frame_y = (y + row) as f32;
            let mask_x =
                transform[[0, 0]] * frame_x + transform[[0, 1]] * frame_y + transform[[0, 2]];
            let mask_y =
                transform[[1, 0]] * frame_x + transform[[1, 1]] * frame_y + transform[[1, 2]];
            sample_mask(mask, mask_x, mask_y)
        });
        Ok(Self {
            original,
            visible,
            x,
            y,
        })
    }

    /// Restore foreground from the original ROI after swapping, mouth blending and enhancement.
    pub(crate) fn restore(&self, image: &mut Array3<u8>) {
        for ((y, x), &alpha) in self.visible.indexed_iter() {
            if alpha >= 1.0 {
                continue;
            }
            for c in 0..3 {
                let original = self.original[[y, x, c]] as f32;
                let processed = image[[self.y + y, self.x + x, c]] as f32;
                image[[self.y + y, self.x + x, c]] = (original + alpha * (processed - original))
                    .round()
                    .clamp(0.0, 255.0) as u8;
            }
        }
    }
}

fn sample_mask(mask: &Array2<f32>, x: f32, y: f32) -> f32 {
    let (height, width) = mask.dim();
    if x < 0.0 || y < 0.0 || x > (width - 1) as f32 || y > (height - 1) as f32 {
        return 0.0;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let dx = x - x0 as f32;
    let dy = y - y0 as f32;
    let top = mask[[y0, x0]] * (1.0 - dx) + mask[[y0, x1]] * dx;
    let bottom = mask[[y1, x0]] * (1.0 - dx) + mask[[y1, x1]] * dx;
    (top * (1.0 - dy) + bottom * dy).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    #[ignore = "requires XSEG_MODEL_PATH and an ONNX Runtime library"]
    fn xseg_model_accepts_synthetic_input() {
        let model =
            std::env::var("XSEG_MODEL_PATH").expect("Set XSEG_MODEL_PATH to the XSeg ONNX file");
        let mut occluder = FaceOccluder::new(&model).unwrap();
        let face = Array4::from_shape_fn((1, 3, 256, 256), |(_, c, y, x)| {
            ((x + y + c * 30) % 256) as f32 / 255.0
        });
        let mask = occluder.visible_mask(&face).unwrap();
        assert_eq!(mask.dim(), (256, 256));
        assert!(mask
            .iter()
            .all(|value| value.is_finite() && (0.0..=1.0).contains(value)));
    }

    #[test]
    fn xseg_input_uses_bgr_nhwc_without_rescaling() {
        let face = Array4::from_shape_fn((1, 3, 256, 256), |(_, c, _, _)| [0.1, 0.4, 0.9][c]);
        let input = prepare_input(&face).unwrap();
        assert_eq!(input.shape(), &[1, 256, 256, 3]);
        assert_eq!(input[[0, 19, 37, 0]], 0.9);
        assert_eq!(input[[0, 19, 37, 2]], 0.1);
        assert!(prepare_input(&Array4::zeros((1, 3, 128, 128))).is_err());
    }

    #[test]
    fn margin_and_blur_preserve_thin_occluders() {
        let mut visible = Array2::ones((31, 31));
        visible.column_mut(15).fill(0.0);
        let adjusted = adjust_mask(&visible, OcclusionOptions { margin: 2, blur: 3 });
        for y in 0..31 {
            for x in 13..=17 {
                assert_eq!(adjusted[[y, x]], 0.0);
            }
        }
        assert!(adjusted[[15, 12]] > 0.0 && adjusted[[15, 12]] < 1.0);
        assert!(adjusted[[15, 5]] > 0.999);
    }

    #[test]
    fn fully_visible_and_occluded_masks_stay_uniform() {
        for value in [0.0, 1.0] {
            let adjusted = adjust_mask(
                &Array2::from_elem((17, 17), value),
                OcclusionOptions::default(),
            );
            assert!(adjusted.iter().all(|&v| (v - value).abs() < 1e-6));
        }
        assert!(OcclusionOptions {
            margin: 17,
            blur: 0
        }
        .validate()
        .is_err());
        assert!(OcclusionOptions {
            margin: 0,
            blur: usize::MAX
        }
        .validate()
        .is_err());
    }

    #[test]
    fn restoration_keeps_original_objects_after_multiple_processing_steps() {
        let original =
            Array3::from_shape_fn((280, 300, 3), |(y, x, c)| ((x + 3 * y + c) % 256) as u8);
        let mut mask = Array2::ones((256, 256));
        mask.slice_mut(s![80..160, 110..140]).fill(0.0);
        let transform = array![[1.0, 0.0, -20.0], [0.0, 1.0, -10.0]];
        let saved = ProtectedRegion::capture(&original, &mask, &transform).unwrap();
        let mut output = original.clone();
        crate::alignment::paste_back_inplace(
            &mut output,
            &Array4::from_elem((1, 3, 128, 128), 0.8),
            &transform.mapv(|v| v * 0.5),
            128,
        )
        .unwrap();
        crate::alignment::paste_back_inplace(
            &mut output,
            &Array4::from_elem((1, 3, 512, 512), 0.3),
            &transform.mapv(|v| v * 2.0),
            512,
        )
        .unwrap();
        saved.restore(&mut output);
        assert_eq!(
            output.slice(s![90..170, 130..160, ..]),
            original.slice(s![90..170, 130..160, ..])
        );
        assert_ne!(output[[130, 100, 0]], original[[130, 100, 0]]);
        assert_eq!(output[[279, 299, 2]], original[[279, 299, 2]]);
    }

    #[test]
    fn rotated_mask_uses_frame_to_aligned_coordinates() {
        let original = Array3::from_elem((256, 256, 3), 17);
        let mask = Array2::from_shape_fn((256, 256), |(_, x)| if x < 128 { 0.0 } else { 1.0 });
        let transform = array![[0.0, -1.0, 255.0], [1.0, 0.0, 0.0]];
        let saved = ProtectedRegion::capture(&original, &mask, &transform).unwrap();
        let mut output = Array3::from_elem((256, 256, 3), 200);
        saved.restore(&mut output);
        assert_eq!(output[[200, 100, 0]], 17);
        assert_eq!(output[[50, 100, 0]], 200);
    }

    #[test]
    fn mask_sampling_interpolates_and_protects_outside_crop() {
        let mask = array![[0.0, 1.0], [0.0, 1.0]];
        assert_eq!(sample_mask(&mask, 0.25, 0.5), 0.25);
        assert_eq!(sample_mask(&mask, -0.1, 0.5), 0.0);
        assert_eq!(sample_mask(&mask, 2.0, 0.0), 0.0);
    }
}
