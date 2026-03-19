//! GET /api/crops/{session_id}/{role}/{filename} - Serve face crop images

use actix_web::{web, HttpResponse};
use crate::error::ErrorResponse;
use crate::state::AppState;

/// Serve a face crop image
#[utoipa::path(
    get,
    tag = "Crops",
    path = "/api/crops/{session_id}/{role}/{filename}",
    params(
        ("session_id" = String, Path, description = "Session identifier", example = "a1b2c3d4"),
        ("role" = String, Path, description = "Face role: source, target, or cluster", example = "source"),
        ("filename" = String, Path, description = "Crop filename", example = "face_0.jpg"),
    ),
    responses(
        (status = 200, description = "JPEG image", content_type = "image/jpeg"),
        (status = 400, description = "Invalid parameters", body = ErrorResponse),
        (status = 404, description = "Crop not found", body = ErrorResponse)
    )
)]
pub async fn serve_crop(
    state: web::Data<AppState>,
    path: web::Path<(String, String, String)>,
) -> HttpResponse {
    let (session_id, role, filename) = path.into_inner();

    if !["source", "target", "cluster"].contains(&role.as_str()) {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error_text: format!("Invalid role: {}", role),
        });
    }

    // Prevent directory traversal
    if session_id.contains("..") || filename.contains("..") {
        return HttpResponse::BadRequest().json(ErrorResponse {
            error_text: "Invalid path".to_string(),
        });
    }

    let file_path = format!("{}/{}/{}/{}", state.tmp_dir, session_id, role, filename);

    match std::fs::read(&file_path) {
        Ok(data) => HttpResponse::Ok()
            .content_type("image/jpeg")
            .body(data),
        Err(_) => HttpResponse::NotFound().json(ErrorResponse {
            error_text: format!("Crop not found: {}/{}/{}", session_id, role, filename),
        }),
    }
}
