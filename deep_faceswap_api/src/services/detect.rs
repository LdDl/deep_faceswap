//! POST /api/detect - Detect faces in source and target images
//!
//! Always returns ALL detected faces (multi-face is forced in the API).
//! Unlike the CLI which has a --multi-face flag and falls back to highest-score,
//! the API always exposes every face and lets the UI handle mapping.

use crate::error::ErrorResponse;
use crate::state::AppState;
use actix_web::{web, HttpRequest, HttpResponse};
use deep_faceswap_core::multi_face::{save_face_crops_from_infos_to, save_face_crops_to};
use deep_faceswap_core::types::{BBox, SourceFaceInfo};
use deep_faceswap_core::utils::{image as img_io, rgb};
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// Face detection request
#[derive(Deserialize, Serialize, ToSchema)]
pub struct DetectRequest {
    /// Paths to source face images
    #[schema(example = json!(["/home/user/source.jpg"]))]
    pub source_paths: Vec<String>,
    /// Path to target image
    #[schema(example = "/home/user/target.jpg")]
    pub target_path: String,
}

/// Information about a single detected face
#[derive(Serialize, ToSchema)]
pub struct FaceInfo {
    #[schema(example = 0)]
    pub index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(example = 0)]
    pub source_image_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(example = "source.jpg")]
    pub source_filename: Option<String>,
    pub bbox: BBox,
    #[schema(example = 0.95)]
    pub det_score: f32,
    #[schema(example = "/api/crops/a1b2c3d4/source/source_face_0.jpg")]
    pub crop_url: String,
}

/// Face detection response
#[derive(Serialize, ToSchema)]
pub struct DetectResponse {
    #[schema(example = "a1b2c3d4")]
    pub session_id: String,
    pub source_faces: Vec<FaceInfo>,
    pub target_faces: Vec<FaceInfo>,
}

/// Detect faces in source and target images
#[utoipa::path(
    post,
    tag = "Detection",
    path = "/api/detect",
    request_body = DetectRequest,
    responses(
        (status = 200, description = "Faces detected", body = DetectResponse),
        (status = 400, description = "Invalid input", body = ErrorResponse),
        (status = 500, description = "Internal error", body = ErrorResponse)
    )
)]
pub async fn detect_faces(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    req: web::Json<DetectRequest>,
) -> HttpResponse {
    let method = http_req.method().to_string();
    let route = http_req.path().to_string();
    let state = state.into_inner();
    let req = req.into_inner();
    let req_body = serde_json::to_string(&req).unwrap_or_default();

    let result = tokio::task::spawn_blocking(move || {
        let session_id = uuid::Uuid::new_v4().to_string()[..8].to_string();
        let crop_base = format!("{}/{}", state.tmp_dir, session_id);

        let mut detector = state.models.detector.lock().unwrap();

        // Detect faces in all source images
        let mut source_images = Vec::new();
        let mut all_source_faces: Vec<SourceFaceInfo> = Vec::new();

        for (img_idx, path) in req.source_paths.iter().enumerate() {
            let img = match img_io::load_image(path) {
                Ok(img) => img,
                Err(e) => {
                    return Err(format!("Cannot load source '{}': {}", path, e));
                }
            };
            let rgb_img = img_io::to_rgb8(&img);
            let array = rgb::rgb_to_array3(&rgb_img);

            let faces = match detector.detect(&array, 0.5, 0.4) {
                Ok(f) => f,
                Err(e) => {
                    return Err(format!("Detection failed on '{}': {}", path, e));
                }
            };

            let filename = std::path::Path::new(path.as_str())
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

        // Detect faces in target image
        let target_img = match img_io::load_image(&req.target_path) {
            Ok(img) => img,
            Err(e) => {
                return Err(format!("Cannot load target '{}': {}", req.target_path, e));
            }
        };
        let target_rgb = img_io::to_rgb8(&target_img);
        let target_array = rgb::rgb_to_array3(&target_rgb);

        let target_faces = match detector.detect(&target_array, 0.5, 0.4) {
            Ok(f) => f,
            Err(e) => {
                return Err(format!("Detection failed on target: {}", e));
            }
        };

        // Save crops
        let source_crop_dir = format!("{}/source", crop_base);
        let source_crops = if !all_source_faces.is_empty() {
            save_face_crops_from_infos_to(&all_source_faces, &source_images, &source_crop_dir)
                .map_err(|e| format!("Failed to save source crops: {}", e))?
        } else {
            Vec::new()
        };

        let target_crops = if !target_faces.is_empty() {
            save_face_crops_to(&target_faces, &target_array, false, &crop_base)
                .map_err(|e| format!("Failed to save target crops: {}", e))?
        } else {
            Vec::new()
        };

        // Build response
        let source_face_infos: Vec<FaceInfo> = source_crops
            .iter()
            .enumerate()
            .map(|(i, crop)| {
                let info = &all_source_faces[i];
                FaceInfo {
                    index: i,
                    source_image_index: Some(info.source_image_index),
                    source_filename: Some(info.source_filename.clone()),
                    bbox: info.face.bbox.clone(),
                    det_score: info.face.det_score,
                    crop_url: format!(
                        "/api/crops/{}/source/{}",
                        session_id,
                        std::path::Path::new(&crop.crop_path)
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                    ),
                }
            })
            .collect();

        let target_face_infos: Vec<FaceInfo> = target_crops
            .iter()
            .enumerate()
            .map(|(i, crop)| {
                let face = &target_faces[i];
                FaceInfo {
                    index: i,
                    source_image_index: None,
                    source_filename: None,
                    bbox: face.bbox.clone(),
                    det_score: face.det_score,
                    crop_url: format!(
                        "/api/crops/{}/target/{}",
                        session_id,
                        std::path::Path::new(&crop.crop_path)
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                    ),
                }
            })
            .collect();

        Ok(DetectResponse {
            session_id,
            source_faces: source_face_infos,
            target_faces: target_face_infos,
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
                "Can't detect faces"
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
                "Can't detect faces"
            );
            HttpResponse::InternalServerError().json(ErrorResponse {
                error_text: err_msg,
            })
        }
    }
}
