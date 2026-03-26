//! GET /api/file?path=... - Serve files from the server filesystem
//!
//! Used for previewing source images, target images/videos, and swap results.
//! Respects allowed_dirs security. Supports HTTP Range requests for video streaming.

use crate::error::ErrorResponse;
use crate::state::AppState;
use actix_web::{web, HttpRequest, HttpResponse};
use serde::Deserialize;
use utoipa::IntoParams;

#[derive(Deserialize, IntoParams)]
pub struct FileQuery {
    /// Absolute path to the file on the server
    #[param(example = "/home/user/photos/source.jpg")]
    pub path: String,
}

/// Serve a file from the server filesystem
#[utoipa::path(
    get,
    tag = "Files",
    path = "/api/file",
    params(FileQuery),
    responses(
        (status = 200, description = "File contents with appropriate Content-Type"),
        (status = 400, description = "Invalid path", body = ErrorResponse),
        (status = 403, description = "Path not allowed", body = ErrorResponse),
        (status = 404, description = "File not found", body = ErrorResponse)
    )
)]
pub async fn serve_file(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    query: web::Query<FileQuery>,
) -> Result<HttpResponse, actix_web::Error> {
    let method = http_req.method().to_string();
    let route = http_req.path().to_string();
    let path = &query.path;

    if path.contains("..") {
        let err_msg = "Invalid path".to_string();
        tracing::error!(
            scope = "api",
            method = method.as_str(),
            route = route.as_str(),
            error = err_msg.as_str(),
            "Can't serve file"
        );
        return Ok(HttpResponse::BadRequest().json(ErrorResponse {
            error_text: err_msg,
        }));
    }

    if !state.is_path_allowed(path) {
        let err_msg = format!("Path not allowed: {}", path);
        tracing::error!(
            scope = "api",
            method = method.as_str(),
            route = route.as_str(),
            error = err_msg.as_str(),
            "Can't serve file"
        );
        return Ok(HttpResponse::Forbidden().json(ErrorResponse {
            error_text: err_msg,
        }));
    }

    // NamedFile handles Content-Type detection and Range requests automatically
    let file = actix_files::NamedFile::open_async(path)
        .await
        .map_err(|_| {
            let err_msg = format!("File not found: {}", path);
            tracing::error!(
                scope = "api",
                method = method.as_str(),
                route = route.as_str(),
                error = err_msg.as_str(),
                "Can't serve file"
            );
            actix_web::error::ErrorNotFound(err_msg)
        })?;

    Ok(file.into_response(&http_req))
}
