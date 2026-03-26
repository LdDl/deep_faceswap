//! Route composition and OpenAPI documentation

use actix_files as fs;
use actix_web::web;
use std::env;

mod crops;
mod detect;
mod files;
mod health;
mod jobs;
mod serve_file;
mod swap_image;
mod video;

use utoipa::OpenApi;
use utoipa_rapidoc::RapiDoc;

pub fn init_routes(cfg: &mut web::ServiceConfig) {
    let ui_dir = env::var("UI_DIR").unwrap_or_default();

    cfg.service(
        web::scope("/api")
            .service(RapiDoc::with_openapi("/docs.json", ApiDoc::openapi()))
            .service(RapiDoc::new("/api/docs.json").path("/docs"))
            .route("/health", web::get().to(health::health))
            .service(web::scope("/files").route("", web::get().to(files::list_files)))
            .service(web::scope("/detect").route("", web::post().to(detect::detect_faces)))
            .service(web::scope("/crops").route(
                "/{session_id}/{role}/{filename}",
                web::get().to(crops::serve_crop),
            ))
            .service(
                web::scope("/swap")
                    .route("/image", web::post().to(swap_image::swap_image))
                    .route("/video", web::post().to(video::swap_video)),
            )
            .service(web::scope("/video").route("/analyze", web::post().to(video::analyze_video)))
            .service(web::scope("/jobs").route("/{job_id}", web::get().to(jobs::get_job_status)))
            .route("/file", web::get().to(serve_file::serve_file)),
    );

    // Serve SvelteKit build if UI_DIR is set
    if !ui_dir.is_empty() {
        cfg.service(
            fs::Files::new("/", &ui_dir)
                .index_file("index.html")
                .default_handler(web::to(move || {
                    let ui_dir = env::var("UI_DIR").unwrap_or_default();
                    async move {
                        actix_files::NamedFile::open_async(format!("{}/index.html", ui_dir)).await
                    }
                })),
        );
    }
}

#[derive(OpenApi)]
#[openapi(
    paths(
        health::health,
        files::list_files,
        detect::detect_faces,
        crops::serve_crop,
        swap_image::swap_image,
        video::analyze_video,
        video::swap_video,
        jobs::get_job_status,
        serve_file::serve_file,
    ),
    tags(
        (name = "System", description = "Health check and system status"),
        (name = "Files", description = "Server filesystem browser"),
        (name = "Detection", description = "Face detection in images"),
        (name = "Crops", description = "Serve face crop images"),
        (name = "Swap", description = "Face swap operations"),
        (name = "Video", description = "Video analysis and processing"),
        (name = "Jobs", description = "Background job management"),
    ),
    components(
        schemas(
            crate::error::ErrorResponse,
            crate::jobs::JobState,
            crate::jobs::JobStatus,
            crate::jobs::JobProgress,
            crate::jobs::JobResult,
            crate::services::health::HealthResponse,
            crate::services::files::FilesResponse,
            crate::services::files::FileEntry,
            crate::services::detect::DetectRequest,
            crate::services::detect::DetectResponse,
            crate::services::detect::FaceInfo,
            crate::services::swap_image::SwapImageRequest,
            crate::services::swap_image::SwapImageResponse,
            crate::services::video::VideoAnalyzeRequest,
            crate::services::video::VideoAnalyzeResponse,
            crate::services::video::VideoSwapRequest,
            crate::services::video::VideoSwapResponse,
            crate::services::video::ClusterInfo,
            deep_faceswap_core::types::BBox,
            deep_faceswap_core::types::FaceMapping,
            deep_faceswap_core::types::ClusterMapping,
        ),
    ),
)]
struct ApiDoc;
