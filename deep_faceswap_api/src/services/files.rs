//! GET /api/files - Browse server filesystem

use crate::error::ErrorResponse;
use crate::state::AppState;
use actix_web::{web, HttpRequest, HttpResponse};
use serde::{Deserialize, Serialize};
use utoipa::{IntoParams, ToSchema};

/// File browser query
#[derive(Deserialize, IntoParams)]
pub struct FilesQuery {
    /// Directory path to list
    #[param(example = "/home/user/photos")]
    pub path: String,
}

/// Single file or directory entry
#[derive(Serialize, ToSchema)]
pub struct FileEntry {
    #[schema(example = "photo.jpg")]
    pub name: String,
    #[schema(example = false)]
    pub is_dir: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    #[schema(example = 2345678)]
    pub size: Option<u64>,
}

/// Directory listing response
#[derive(Serialize, ToSchema)]
pub struct FilesResponse {
    #[schema(example = "/home/user/photos")]
    pub path: String,
    pub entries: Vec<FileEntry>,
}

/// Browse server filesystem for file selection
#[utoipa::path(
    get,
    tag = "Files",
    path = "/api/files",
    params(FilesQuery),
    responses(
        (status = 200, description = "Directory listing", body = FilesResponse),
        (status = 400, description = "Path not allowed or not found", body = ErrorResponse)
    )
)]
pub async fn list_files(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    query: web::Query<FilesQuery>,
) -> HttpResponse {
    let method = http_req.method().to_string();
    let route = http_req.path().to_string();
    let query_string = http_req.query_string().to_string();
    let path = &query.path;

    if !state.is_path_allowed(path) {
        let err_msg = format!("Path not allowed: {}", path);
        tracing::error!(
            scope = "api",
            method = method.as_str(),
            route = route.as_str(),
            query = query_string.as_str(),
            error = err_msg.as_str(),
            "Can't list files"
        );
        return HttpResponse::BadRequest().json(ErrorResponse {
            error_text: err_msg,
        });
    }

    let read_dir = match std::fs::read_dir(path) {
        Ok(rd) => rd,
        Err(e) => {
            let err_msg = format!("Cannot read directory '{}': {}", path, e);
            tracing::error!(
                scope = "api",
                method = method.as_str(),
                route = route.as_str(),
                query = query_string.as_str(),
                error = err_msg.as_str(),
                "Can't list files"
            );
            return HttpResponse::BadRequest().json(ErrorResponse {
                error_text: err_msg,
            });
        }
    };

    let mut entries = Vec::new();

    // Parent directory entry
    if let Some(parent) = std::path::Path::new(path).parent() {
        let parent_str = parent.to_string_lossy().to_string();
        if state.is_path_allowed(&parent_str) {
            entries.push(FileEntry {
                name: "..".to_string(),
                is_dir: true,
                size: None,
            });
        }
    }

    for entry in read_dir {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };

        let name = entry.file_name().to_string_lossy().to_string();

        // Skip hidden files
        if name.starts_with('.') {
            continue;
        }

        let metadata = match entry.metadata() {
            Ok(m) => m,
            Err(_) => continue,
        };

        let is_dir = metadata.is_dir();
        let size = if is_dir { None } else { Some(metadata.len()) };

        entries.push(FileEntry { name, is_dir, size });
    }

    // Sort: directories first, then files, alphabetically within each group
    entries.sort_by(|a, b| {
        if a.name == ".." {
            return std::cmp::Ordering::Less;
        }
        if b.name == ".." {
            return std::cmp::Ordering::Greater;
        }
        match (a.is_dir, b.is_dir) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => a.name.to_lowercase().cmp(&b.name.to_lowercase()),
        }
    });

    HttpResponse::Ok().json(FilesResponse {
        path: path.to_string(),
        entries,
    })
}
