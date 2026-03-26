//! GET /api/crops/{session_id}/{role}/{filename} - Serve face crop images

use crate::error::ErrorResponse;
use crate::state::AppState;
use actix_web::{web, HttpRequest, HttpResponse};

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
    http_req: HttpRequest,
    state: web::Data<AppState>,
    path: web::Path<(String, String, String)>,
) -> HttpResponse {
    let method = http_req.method().to_string();
    let route = http_req.path().to_string();
    let (session_id, role, filename) = path.into_inner();

    if !["source", "target", "cluster"].contains(&role.as_str()) {
        let err_msg = format!("Invalid role: {}", role);
        tracing::error!(
            scope = "api",
            method = method.as_str(),
            route = route.as_str(),
            error = err_msg.as_str(),
            "Can't serve crop"
        );
        return HttpResponse::BadRequest().json(ErrorResponse {
            error_text: err_msg,
        });
    }

    // Prevent directory traversal
    if session_id.contains("..") || filename.contains("..") {
        let err_msg = "Invalid path".to_string();
        tracing::error!(
            scope = "api",
            method = method.as_str(),
            route = route.as_str(),
            error = err_msg.as_str(),
            "Can't serve crop"
        );
        return HttpResponse::BadRequest().json(ErrorResponse {
            error_text: err_msg,
        });
    }

    // Look up session-specific tmp_dir, fall back to default
    let base_dir = {
        let dirs = state.session_dirs.lock().unwrap();
        dirs.get(&session_id)
            .cloned()
            .unwrap_or_else(|| state.tmp_dir.clone())
    };
    let file_path = format!("{}/{}/{}/{}", base_dir, session_id, role, filename);

    match std::fs::read(&file_path) {
        Ok(data) => HttpResponse::Ok().content_type("image/jpeg").body(data),
        Err(_) => {
            let err_msg = format!("Crop not found: {}/{}/{}", session_id, role, filename);
            tracing::error!(
                scope = "api",
                method = method.as_str(),
                route = route.as_str(),
                error = err_msg.as_str(),
                "Can't serve crop"
            );
            HttpResponse::NotFound().json(ErrorResponse {
                error_text: err_msg,
            })
        }
    }
}
