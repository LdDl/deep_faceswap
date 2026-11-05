//! GET /api/health

use crate::state::AppState;
use actix_web::{web, HttpResponse};
use serde::Serialize;
use utoipa::ToSchema;

/// Health check response
#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    #[schema(example = "ok")]
    pub status: String,
    #[schema(example = true)]
    pub cuda: bool,
    #[schema(example = true)]
    pub foreground_available: bool,
}

/// Health check and model status
#[utoipa::path(
    get,
    tag = "System",
    path = "/api/health",
    responses(
        (status = 200, description = "Service is healthy", body = HealthResponse)
    )
)]
pub async fn health(state: web::Data<AppState>) -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
        cuda: cfg!(feature = "cuda"),
        foreground_available: state.models.occluder.is_some(),
    })
}
