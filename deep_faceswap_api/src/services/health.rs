//! GET /api/health

use actix_web::HttpResponse;
use serde::Serialize;
use utoipa::ToSchema;

/// Health check response
#[derive(Serialize, ToSchema)]
pub struct HealthResponse {
    #[schema(example = "ok")]
    pub status: String,
    #[schema(example = true)]
    pub cuda: bool,
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
pub async fn health() -> HttpResponse {
    HttpResponse::Ok().json(HealthResponse {
        status: "ok".to_string(),
        cuda: cfg!(feature = "cuda"),
    })
}
