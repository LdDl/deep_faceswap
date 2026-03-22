//! Video endpoints:
//! POST /api/video/analyze - Analyze video (detect faces, cluster)
//! POST /api/swap/video - Start video swap job (async)

use actix_web::{web, HttpRequest, HttpResponse};
use deep_faceswap_core::multi_face::{save_cluster_crops_to, save_face_crops_from_infos_to};
use deep_faceswap_core::types::{ClusterMapping, SourceFaceInfo};
use deep_faceswap_core::utils::{image as img_io, rgb};
use deep_faceswap_core::video::{
    build_cluster_info, build_face_lookup, cluster_faces, extract_frames, scan_frames_for_faces,
};
use deep_faceswap_core::alignment;
use deep_faceswap_core::enhancer::FaceEnhancer;
use deep_faceswap_core::landmark::LandmarkDetector;
use deep_faceswap_core::swap::swap_video_frames_with_mappings;
use deep_faceswap_core::video;
use serde::{Deserialize, Serialize};
use utoipa::ToSchema;
use uuid::Uuid;
use std::time::Instant;
use std::path::Path;
use std::fs;
use crate::error::ErrorResponse;
use crate::jobs::{JobProgress, JobResult, JobState, JobStatus};
use crate::services::detect::FaceInfo;
use crate::state::AppState;

/// Cluster information from video analysis
#[derive(Serialize, ToSchema)]
pub struct ClusterInfo {
    #[schema(example = 0)]
    pub cluster_id: usize,
    #[schema(example = 245)]
    pub frame_count: usize,
    #[schema(example = "/api/crops/v1b2c3d4/cluster/cluster_0.jpg")]
    pub crop_url: String,
}

/// Video analysis request
#[derive(Deserialize, Serialize, ToSchema)]
pub struct VideoAnalyzeRequest {
    /// Paths to source face images
    #[schema(example = json!(["/home/user/source.jpg"]))]
    pub source_paths: Vec<String>,
    /// Path to target video
    #[schema(example = "/home/user/video.mp4")]
    pub target_video_path: String,
    /// Custom directory for temporary files (frames, crops). Falls back to server default.
    #[serde(default)]
    pub tmp_dir: Option<String>,
}

/// Video analysis response with source faces and target clusters
#[derive(Serialize, ToSchema)]
pub struct VideoAnalyzeResponse {
    #[schema(example = "v1b2c3d4")]
    pub session_id: String,
    pub source_faces: Vec<FaceInfo>,
    pub clusters: Vec<ClusterInfo>,
    #[schema(example = 300)]
    pub total_frames: usize,
    #[schema(example = 45.2)]
    pub elapsed_s: f64,
}

/// Video swap request with cluster mappings
#[derive(Deserialize, Serialize, ToSchema)]
pub struct VideoSwapRequest {
    /// Session ID from video analysis
    #[schema(example = "v1b2c3d4")]
    pub session_id: String,
    /// Paths to source face images
    #[schema(example = json!(["/home/user/source.jpg"]))]
    pub source_paths: Vec<String>,
    /// Path to target video
    #[schema(example = "/home/user/video.mp4")]
    pub target_video_path: String,
    /// Path to save output video
    #[schema(example = "/home/user/output.mp4")]
    pub output_path: String,
    /// Mappings from source faces to target clusters
    pub cluster_mappings: Vec<ClusterMapping>,
    /// Enable face enhancement with GFPGAN
    #[serde(default)]
    #[schema(example = true)]
    pub enhance: bool,
    /// Enable mouth mask
    #[serde(default)]
    #[schema(example = false)]
    pub mouth_mask: bool,
    /// Custom directory for temporary files. Falls back to server default.
    #[serde(default)]
    pub tmp_dir: Option<String>,
}

/// Video swap job creation response
#[derive(Serialize, ToSchema)]
pub struct VideoSwapResponse {
    #[schema(example = "job-abc123")]
    pub job_id: String,
    #[schema(example = "queued")]
    pub status: String,
}

/// Analyze video: extract frames, scan faces, cluster
#[utoipa::path(
    post,
    tag = "Video",
    path = "/api/video/analyze",
    request_body = VideoAnalyzeRequest,
    responses(
        (status = 200, description = "Video analyzed", body = VideoAnalyzeResponse),
        (status = 400, description = "Invalid input", body = ErrorResponse),
        (status = 500, description = "Internal error", body = ErrorResponse)
    )
)]
pub async fn analyze_video(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    req: web::Json<VideoAnalyzeRequest>,
) -> HttpResponse {
    let method = http_req.method().to_string();
    let route = http_req.path().to_string();
    let state = state.into_inner();
    let req = req.into_inner();
    let req_body = serde_json::to_string(&req).unwrap_or_default();

    let result = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let session_id = Uuid::new_v4().to_string()[..8].to_string();
        let base_tmp = req.tmp_dir.as_deref()
            .filter(|s| !s.is_empty())
            .unwrap_or(&state.tmp_dir);
        let crop_base = format!("{}/{}", base_tmp, session_id);

        let mut detector = state.models.detector.lock().unwrap();
        let mut recognizer = state.models.recognizer.lock().unwrap();

        // Load and detect source faces
        let mut source_images = Vec::new();
        let mut all_source_faces: Vec<SourceFaceInfo> = Vec::new();

        for (img_idx, path) in req.source_paths.iter().enumerate() {
            let img = img_io::load_image(path)
                .map_err(|e| format!("Cannot load source '{}': {}", path, e))?;
            let rgb_img = img_io::to_rgb8(&img);
            let array = rgb::rgb_to_array3(&rgb_img);
            let faces = detector
                .detect(&array, 0.5, 0.4)
                .map_err(|e| format!("Detection failed: {}", e))?;

            let filename = Path::new(path.as_str())
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

        if all_source_faces.is_empty() {
            return Err("No faces detected in source images".to_string());
        }

        // Save source crops
        let source_crop_dir = format!("{}/source", crop_base);
        let source_crops =
            save_face_crops_from_infos_to(&all_source_faces, &source_images, &source_crop_dir)
                .map_err(|e| format!("Failed to save source crops: {}", e))?;

        // Extract frames
        let frames_dir = format!("{}/video_frames", crop_base);
        fs::create_dir_all(&frames_dir)
            .map_err(|e| format!("Cannot create frames dir: {}", e))?;
        let frame_paths = extract_frames(&req.target_video_path, &frames_dir)
            .map_err(|e| format!("Frame extraction failed: {}", e))?;
        let total_frames = frame_paths.len();

        // Scan all frames for faces and cluster
        let mut face_records = scan_frames_for_faces(&frame_paths, &mut detector, &mut recognizer)
            .map_err(|e| format!("Face scanning failed: {}", e))?;
        let centroids = cluster_faces(&mut face_records, 10)
            .map_err(|e| format!("Clustering failed: {}", e))?;
        let cluster_infos = build_cluster_info(&face_records, &centroids)
            .map_err(|e| format!("Cluster info failed: {}", e))?;

        // Save cluster crops
        let cluster_crop_dir = format!("{}/cluster", crop_base);
        let cluster_crops =
            save_cluster_crops_to(&cluster_infos, &frame_paths, &cluster_crop_dir)
                .map_err(|e| format!("Failed to save cluster crops: {}", e))?;

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
                        Path::new(&crop.crop_path)
                            .file_name()
                            .unwrap()
                            .to_string_lossy()
                    ),
                }
            })
            .collect();

        let clusters: Vec<ClusterInfo> = cluster_crops
            .iter()
            .map(|crop| ClusterInfo {
                cluster_id: crop.cluster_id,
                frame_count: crop.frame_count,
                crop_url: format!(
                    "/api/crops/{}/cluster/{}",
                    session_id,
                    Path::new(&crop.crop_path)
                        .file_name()
                        .unwrap()
                        .to_string_lossy()
                ),
            })
            .collect();

        Ok(VideoAnalyzeResponse {
            session_id,
            source_faces: source_face_infos,
            clusters,
            total_frames,
            elapsed_s: start.elapsed().as_secs_f64(),
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
                "Can't analyze video"
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
                "Can't analyze video"
            );
            HttpResponse::InternalServerError().json(ErrorResponse { error_text: err_msg })
        }
    }
}

/// Start async video swap job
#[utoipa::path(
    post,
    tag = "Video",
    path = "/api/swap/video",
    request_body = VideoSwapRequest,
    responses(
        (status = 202, description = "Job created", body = VideoSwapResponse),
        (status = 500, description = "Internal error", body = ErrorResponse)
    )
)]
pub async fn swap_video(
    http_req: HttpRequest,
    state: web::Data<AppState>,
    req: web::Json<VideoSwapRequest>,
) -> HttpResponse {
    let method = http_req.method().to_string();
    let route = http_req.path().to_string();
    let state = state.into_inner();
    let req = req.into_inner();
    let req_body = serde_json::to_string(&req).unwrap_or_default();

    let job_id = Uuid::new_v4().to_string()[..8].to_string();

    // Create job entry
    {
        let mut jobs = state.jobs.lock().unwrap();
        jobs.insert(job_id.clone(), JobState::new_queued(job_id.clone()));
    }

    let state_clone = state.clone();
    let job_id_clone = job_id.clone();
    let log_method = method;
    let log_route = route;
    let log_body = req_body;

    tokio::task::spawn_blocking(move || {
        // Update status to running
        {
            let mut jobs = state_clone.jobs.lock().unwrap();
            if let Some(job) = jobs.get_mut(&job_id_clone) {
                job.status = JobStatus::Running;
                job.progress.stage = "initializing".to_string();
            }
        }

        let result: Result<(), String> = (|| {
            let mut detector = state_clone.models.detector.lock().unwrap();
            let mut recognizer = state_clone.models.recognizer.lock().unwrap();
            let mut swapper = state_clone.models.swapper.lock().unwrap();
            let mut enhancer = state_clone.models.enhancer.lock().unwrap();
            let mut landmark_detector = state_clone.models.landmark_detector.lock().unwrap();

            // Conditionally disable enhancer/landmark per request
            let mut no_enhancer: Option<FaceEnhancer> = None;
            let mut no_landmark: Option<LandmarkDetector> = None;
            let enhancer_ref: &mut Option<_> = if req.enhance { &mut *enhancer } else { &mut no_enhancer };
            let landmark_ref: &mut Option<_> = if req.mouth_mask { &mut *landmark_detector } else { &mut no_landmark };

            let base_tmp = req.tmp_dir.as_deref()
                .filter(|s| !s.is_empty())
                .unwrap_or(&state_clone.tmp_dir);
            let session_dir = format!("{}/{}", base_tmp, req.session_id);

            // Load source images and detect faces
            let mut source_images = Vec::new();
            let mut all_source_faces: Vec<SourceFaceInfo> = Vec::new();

            for (img_idx, path) in req.source_paths.iter().enumerate() {
                let img = img_io::load_image(path)
                    .map_err(|e| format!("Cannot load source: {}", e))?;
                let rgb_img = img_io::to_rgb8(&img);
                let array = rgb::rgb_to_array3(&rgb_img);
                let faces = detector
                    .detect(&array, 0.5, 0.4)
                    .map_err(|e| format!("Detection failed: {}", e))?;

                let filename = Path::new(path.as_str())
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

            if all_source_faces.is_empty() {
                return Err("No faces detected in source images".to_string());
            }

            // Extract source embeddings
            let mut source_embeddings = Vec::new();
            for info in &all_source_faces {
                let source_img = &source_images[info.source_image_index];
                let aligned =
                    alignment::align_face(source_img, &info.face, 112)
                        .map_err(|e| format!("Alignment failed: {}", e))?;
                let embedding = recognizer
                    .extract_embedding(&aligned.aligned_image)
                    .map_err(|e| format!("Embedding failed: {}", e))?;
                source_embeddings.push(embedding);
            }

            // Re-use frames from analysis session
            let frames_dir = format!("{}/video_frames", session_dir);
            let processed_dir = format!("{}/video_frames_processed", session_dir);
            fs::create_dir_all(&processed_dir)
                .map_err(|e| format!("Cannot create processed dir: {}", e))?;

            // Collect sorted frame paths
            let mut frame_paths: Vec<String> = Vec::new();
            let mut entries: Vec<_> = fs::read_dir(&frames_dir)
                .map_err(|e| format!("Cannot read frames dir: {}", e))?
                .filter_map(|e| e.ok())
                .collect();
            entries.sort_by_key(|e| e.file_name());
            for entry in entries {
                frame_paths.push(entry.path().to_string_lossy().to_string());
            }

            let total_frames = frame_paths.len();

            // Update progress
            {
                let mut jobs = state_clone.jobs.lock().unwrap();
                if let Some(job) = jobs.get_mut(&job_id_clone) {
                    job.progress = JobProgress {
                        stage: "scanning_faces".to_string(),
                        current: 0,
                        total: total_frames,
                    };
                }
            }

            // Scan frames and cluster
            let mut face_records =
                scan_frames_for_faces(&frame_paths, &mut detector, &mut recognizer)
                    .map_err(|e| format!("Face scanning failed: {}", e))?;
            let _centroids = cluster_faces(&mut face_records, 10)
                .map_err(|e| format!("Clustering failed: {}", e))?;
            let face_lookup = build_face_lookup(&face_records);

            // Update progress stage
            {
                let mut jobs = state_clone.jobs.lock().unwrap();
                if let Some(job) = jobs.get_mut(&job_id_clone) {
                    job.progress.stage = "processing_frames".to_string();
                }
            }

            // Create progress callback that updates job state
            let state_for_cb = state_clone.clone();
            let job_id_for_cb = job_id_clone.clone();
            let progress_cb = move |current: usize, total: usize| {
                let mut jobs = state_for_cb.jobs.lock().unwrap();
                if let Some(job) = jobs.get_mut(&job_id_for_cb) {
                    job.progress.current = current;
                    job.progress.total = total;
                }
            };

            swap_video_frames_with_mappings(
                &frame_paths,
                &processed_dir,
                &req.cluster_mappings,
                &face_lookup,
                &all_source_faces,
                &source_images,
                &source_embeddings,
                &mut swapper,
                enhancer_ref,
                landmark_ref,
                req.mouth_mask,
                Some(&progress_cb),
            )
            .map_err(|e| format!("Frame processing failed: {}", e))?;

            // Encode video
            {
                let mut jobs = state_clone.jobs.lock().unwrap();
                if let Some(job) = jobs.get_mut(&job_id_clone) {
                    job.progress.stage = "encoding".to_string();
                }
            }

            video::encode_video(
                &processed_dir,
                &req.output_path,
                &req.target_video_path,
            )
            .map_err(|e| format!("Video encoding failed: {}", e))?;

            // Cleanup processed frames
            let _ = fs::remove_dir_all(&processed_dir);

            Ok(())
        })();

        // Update job with final status
        let mut jobs = state_clone.jobs.lock().unwrap();
        if let Some(job) = jobs.get_mut(&job_id_clone) {
            match result {
                Ok(()) => {
                    job.status = JobStatus::Completed;
                    job.progress.stage = "completed".to_string();
                    job.result = Some(JobResult {
                        output_path: req.output_path,
                    });
                }
                Err(e) => {
                    tracing::error!(
                        scope = "api",
                        method = log_method.as_str(),
                        route = log_route.as_str(),
                        body = log_body.as_str(),
                        job_id = job_id_clone.as_str(),
                        error = e.as_str(),
                        "Can't swap video"
                    );
                    job.status = JobStatus::Failed;
                    job.progress.stage = "failed".to_string();
                    job.error = Some(e);
                }
            }
        }
    });

    HttpResponse::Accepted().json(VideoSwapResponse {
        job_id,
        status: "queued".to_string(),
    })
}
