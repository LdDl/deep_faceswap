//! GET /api/jobs/{job_id} - Poll job status

use actix_web::{web, HttpResponse};
use crate::error::ErrorResponse;
use crate::jobs::JobState;
use crate::state::AppState;

/// Poll job status and progress
#[utoipa::path(
    get,
    tag = "Jobs",
    path = "/api/jobs/{job_id}",
    params(
        ("job_id" = String, Path, description = "Job identifier", example = "a1b2c3d4"),
    ),
    responses(
        (status = 200, description = "Job status", body = JobState),
        (status = 404, description = "Job not found", body = ErrorResponse)
    )
)]
pub async fn get_job_status(
    state: web::Data<AppState>,
    path: web::Path<String>,
) -> HttpResponse {
    let job_id = path.into_inner();
    let jobs = state.jobs.lock().unwrap();

    match jobs.get(&job_id) {
        Some(job) => HttpResponse::Ok().json(job),
        None => HttpResponse::NotFound().json(ErrorResponse {
            error_text: format!("Job not found: {}", job_id),
        }),
    }
}
