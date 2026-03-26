//! Error response type shared across all services

use serde::Serialize;
use utoipa::ToSchema;

/// Standard error response returned by all endpoints
#[derive(Debug, Serialize, ToSchema)]
pub struct ErrorResponse {
    #[schema(example = "No faces detected in source image")]
    pub error_text: String,
}
