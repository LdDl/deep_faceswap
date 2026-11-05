//! POST /api/swap/image - Execute face swap on still image

use crate::error::ErrorResponse;
use crate::state::AppState;
use actix_web::{web, HttpRequest, HttpResponse};
use deep_faceswap_core::enhancer::FaceEnhancer;
use deep_faceswap_core::landmark::LandmarkDetector;
use deep_faceswap_core::occlusion::{ForegroundProtection, OcclusionOptions};
use deep_faceswap_core::swap::swap_with_mappings_with_foreground;
use deep_faceswap_core::types::{FaceMapping, SourceFaceInfo};
use deep_faceswap_core::utils::{image as img_io, rgb};
use serde::{Deserialize, Serialize};
use std::path::Path;
use utoipa::ToSchema;

/// Image swap request with explicit face mappings
#[derive(Deserialize, Serialize, ToSchema)]
pub struct SwapImageRequest {
    /// Paths to source face images
    #[schema(example = json!(["/home/user/source.jpg"]))]
    pub source_paths: Vec<String>,
    /// Path to target image
    #[schema(example = "/home/user/target.jpg")]
    pub target_path: String,
    /// Path to save output image
    #[schema(example = "/home/user/output.jpg")]
    pub output_path: String,
    /// Face mappings: which source face goes to which target face
    pub mappings: Vec<FaceMapping>,
    /// Enable face enhancement with GFPGAN
    #[serde(default)]
    #[schema(example = true)]
    pub enhance: bool,
    /// Enable mouth mask to preserve target's mouth expression
    #[serde(default)]
    #[schema(example = false)]
    pub mouth_mask: bool,
    /// Preserve objects covering the face, including hands and food
    #[serde(default)]
    pub preserve_foreground: bool,
    /// Foreground margin and edge softness in aligned face pixels
    #[serde(default)]
    pub foreground_options: OcclusionOptions,
}

/// Image swap result
#[derive(Serialize, ToSchema)]
pub struct SwapImageResponse {
    #[schema(example = "/home/user/output.jpg")]
    pub output_path: String,
    #[schema(example = 2.5)]
    pub elapsed_s: f64,
    #[schema(example = 2)]
    pub faces_swapped: usize,
}

/// Execute face swap on a still image with explicit mappings
#[utoipa::path(
    post,
    tag = "Swap",
    path = "/api/swap/image",
    request_body = SwapImageRequest,
    responses(
        (status = 200, description = "Swap completed", body = SwapImageResponse),
        (status = 400, description = "Invalid input or mapping", body = ErrorResponse),
        (status = 500, description = "Internal error", body = ErrorResponse)
    )
)]
pub async fn swap_image(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    req: web::Json<SwapImageRequest>,
) -> HttpResponse {
    let method = http_req.method().to_string();
    let route = http_req.path().to_string();
    let state = state.into_inner();
    let req = req.into_inner();
    let req_body = serde_json::to_string(&req).unwrap_or_default();

    if let Err(error) = req.foreground_options.validate() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error_text: error.to_string(),
        });
    }
    if req.preserve_foreground && state.models.occluder.is_none() {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error_text: "Foreground preservation is not available on this server".to_string(),
        });
    }

    let result = tokio::task::spawn_blocking(move || {
        let mut detector = state.models.detector.lock().unwrap();
        let mut recognizer = state.models.recognizer.lock().unwrap();
        let mut swapper = state.models.swapper.lock().unwrap();
        let mut enhancer = state.models.enhancer.lock().unwrap();
        let mut landmark_detector = state.models.landmark_detector.lock().unwrap();
        let mut occluder = if req.preserve_foreground {
            Some(
                state
                    .models
                    .occluder
                    .as_ref()
                    .ok_or_else(|| {
                        "Foreground preservation is not available on this server".to_string()
                    })?
                    .lock()
                    .map_err(|_| "Face occluder is unavailable".to_string())?,
            )
        } else {
            None
        };
        let mut foreground = occluder
            .as_deref_mut()
            .map(|occluder| ForegroundProtection {
                occluder,
                options: req.foreground_options,
            });

        // Conditionally disable enhancer/landmark per request
        let mut no_enhancer: Option<FaceEnhancer> = None;
        let mut no_landmark: Option<LandmarkDetector> = None;
        let enhancer_ref: &mut Option<_> = if req.enhance {
            &mut *enhancer
        } else {
            &mut no_enhancer
        };
        let landmark_ref: &mut Option<_> = if req.mouth_mask {
            &mut *landmark_detector
        } else {
            &mut no_landmark
        };

        // Load and detect source faces
        let mut source_images = Vec::new();
        let mut all_source_faces: Vec<SourceFaceInfo> = Vec::new();

        for (img_idx, path) in req.source_paths.iter().enumerate() {
            let img = img_io::load_image(path)
                .map_err(|e| format!("Cannot load source '{}': {}", path, e))?;
            let rgb_img = img_io::to_rgb8(&img);
            let array = rgb::rgb_to_array3(&rgb_img);
            let faces = detector
                .detect(&array, 0.5, 0.4)
                .map_err(|e| format!("Detection failed on '{}': {}", path, e))?;

            let filename = Path::new(path.as_str())
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path);

            for face in faces {
                all_source_faces.push(SourceFaceInfo {
                    face,
                    source_image_index: img_idx,
                    source_filename: filename.to_string(),
                });
            }
            source_images.push(array);
        }

        if all_source_faces.is_empty() {
            return Err("No faces detected in source images".to_string());
        }

        // Load and detect target faces
        let target_img = img_io::load_image(&req.target_path)
            .map_err(|e| format!("Cannot load target '{}': {}", req.target_path, e))?;
        let target_rgb = img_io::to_rgb8(&target_img);
        let target_array = rgb::rgb_to_array3(&target_rgb);
        let target_faces = detector
            .detect(&target_array, 0.5, 0.4)
            .map_err(|e| format!("Detection failed on target: {}", e))?;

        if target_faces.is_empty() {
            return Err("No faces detected in target image".to_string());
        }

        // Validate mappings
        for mapping in &req.mappings {
            if mapping.source_idx >= all_source_faces.len() {
                return Err(format!(
                    "Invalid source_idx {}: only {} source faces detected",
                    mapping.source_idx,
                    all_source_faces.len()
                ));
            }
            if mapping.target_idx >= target_faces.len() {
                return Err(format!(
                    "Invalid target_idx {}: only {} target faces detected",
                    mapping.target_idx,
                    target_faces.len()
                ));
            }
        }

        let faces_swapped = req.mappings.len();
        let mut result_image = target_array.clone();

        let elapsed_s = swap_with_mappings_with_foreground(
            &all_source_faces,
            &target_faces,
            &source_images,
            &mut result_image,
            &req.mappings,
            &mut recognizer,
            &mut swapper,
            enhancer_ref,
            landmark_ref,
            req.mouth_mask,
            foreground.as_mut(),
        )
        .map_err(|e| format!("Swap failed: {}", e))?;

        // Save result
        let result_rgb = rgb::array3_to_rgb(&result_image);
        img_io::save_image(&result_rgb, &req.output_path)
            .map_err(|e| format!("Cannot save output: {}", e))?;

        Ok(SwapImageResponse {
            output_path: req.output_path,
            elapsed_s,
            faces_swapped,
        })
    })
    .await;

    match result {
        Ok(Ok(response)) => HttpResponse::Ok().json(response),
        Ok(Err(msg)) => {
            tracing::error!(
                scope = "api",
                method = method.as_str(),
                route = route.as_str(),
                body = req_body.as_str(),
                error = msg.as_str(),
                "Can't swap image"
            );
            HttpResponse::BadRequest().json(ErrorResponse { error_text: msg })
        }
        Err(e) => {
            let err_msg = format!("Task failed: {}", e);
            tracing::error!(
                scope = "api",
                method = method.as_str(),
                route = route.as_str(),
                body = req_body.as_str(),
                error = err_msg.as_str(),
                "Can't swap image"
            );
            HttpResponse::InternalServerError().json(ErrorResponse {
                error_text: err_msg,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_image_requests_keep_foreground_preservation_disabled() {
        let request: SwapImageRequest = serde_json::from_value(serde_json::json!({
            "source_paths": ["source.jpg"],
            "target_path": "target.jpg",
            "output_path": "output.jpg",
            "mappings": [{"source_idx": 0, "target_idx": 0}]
        }))
        .unwrap();
        assert!(!request.preserve_foreground);
        assert_eq!(request.foreground_options.margin, 2);
        assert_eq!(request.foreground_options.blur, 3);
    }
}
