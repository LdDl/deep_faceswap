//! Video face analysis and clustering
//!
//! Scans all video frames to detect faces, extracts ArcFace embeddings,
//! clusters them with K-means (auto-selecting K via silhouette score),
//! and builds lookup structures for per-frame processing.
//!
//! Pipeline:
//! 1. `scan_frames_for_faces` - detect faces and extract embeddings from all frames
//! 2. `cluster_faces` - group embeddings into identity clusters via K-means
//! 3. `build_cluster_info` - summarize clusters for interactive source-to-cluster mapping
//! 4. `build_face_lookup` - build (frame_idx, cluster_id) -> faces lookup table

use crate::clustering::{find_nearest_centroid, kmeans_cluster, select_optimal_k};
use crate::detection::FaceDetector;
use crate::recognition::FaceRecognizer;
use crate::types::{DetectedFace, FaceSwapError, Result};
use crate::utils::{image as img_io, rgb};
use crate::{log_additional, log_main};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Single face detected in a video frame, with its embedding and cluster assignment
#[derive(Clone, Serialize, Deserialize)]
pub struct FaceRecord {
    /// Index of the frame where this face was detected
    pub frame_idx: usize,
    /// Detection result (bounding box, landmarks, score)
    pub face: DetectedFace,
    /// ArcFace embedding vector (512-dimensional, L2-normalized)
    pub embedding: Vec<f32>,
    /// Cluster assignment after K-means (None before clustering)
    pub cluster_id: Option<usize>,
}

/// Summary of a face cluster for interactive source-to-cluster mapping
pub struct ClusterInfo {
    /// Cluster index (0-based)
    pub cluster_id: usize,
    /// Cluster centroid embedding (mean of all assigned face embeddings)
    pub centroid: Vec<f32>,
    /// Best example face from this cluster (highest detection score)
    pub example_face: DetectedFace,
    /// Number of frames where this cluster's face appears
    pub frame_count: usize,
    /// Frame index of the example face (for loading the crop image)
    pub example_frame_idx: usize,
}

/// Scan all video frames and extract face embeddings
///
/// # Arguments
/// * `frame_paths` - Paths to all video frames
/// * `detector` - Face detector
/// * `recognizer` - Face recognizer for embedding extraction
///
/// # Returns
/// Vector of face records with embeddings
pub fn scan_frames_for_faces(
    frame_paths: &[String],
    detector: &mut FaceDetector,
    recognizer: &mut FaceRecognizer,
) -> Result<Vec<FaceRecord>> {
    log_main!(
        "video_analysis",
        "Scanning all frames for faces",
        total_frames = frame_paths.len()
    );

    let mut face_records = Vec::new();
    let log_interval = std::cmp::max(10, frame_paths.len() / 10);

    for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
        let frame_img = img_io::load_image(frame_path)?;
        let frame_rgb = img_io::to_rgb8(&frame_img);
        let frame_array = rgb::rgb_to_array3(&frame_rgb);

        let faces = detector.detect(&frame_array, 0.5, 0.4)?;

        for face in faces {
            log_additional!(
                "video_analysis",
                "Extracting embedding for face",
                frame_idx = frame_idx
            );

            let aligned_face = crate::alignment::align_face(&frame_array, &face, 112)?;
            let embedding = recognizer.extract_embedding(&aligned_face.aligned_image)?;

            face_records.push(FaceRecord {
                frame_idx,
                face,
                embedding,
                cluster_id: None,
            });
        }

        if (frame_idx + 1) % log_interval == 0 {
            log_main!(
                "video_analysis",
                "Scanning frames",
                processed = frame_idx + 1,
                total = frame_paths.len(),
                faces_found = face_records.len()
            );
        }
    }

    log_main!(
        "video_analysis",
        "Scan complete",
        total_faces = face_records.len()
    );

    if face_records.is_empty() {
        return Err(FaceSwapError::ProcessingError(
            "No faces found in any video frame".to_string(),
        ));
    }

    Ok(face_records)
}

/// Cluster face records using K-means
///
/// # Arguments
/// * `face_records` - Face records with embeddings
/// * `max_k` - Maximum number of clusters
///
/// # Returns
/// Updated face records with cluster assignments and centroids
pub fn cluster_faces(face_records: &mut Vec<FaceRecord>, max_k: usize) -> Result<Array2<f32>> {
    log_main!(
        "video_analysis",
        "Clustering faces",
        total_faces = face_records.len()
    );

    if face_records.is_empty() {
        return Err(FaceSwapError::ProcessingError(
            "No faces to cluster".to_string(),
        ));
    }

    // Convert embeddings to ndarray
    let n_faces = face_records.len();
    let emb_dim = face_records[0].embedding.len();
    let mut embeddings = Array2::<f32>::zeros((n_faces, emb_dim));

    for (i, record) in face_records.iter().enumerate() {
        for (j, &val) in record.embedding.iter().enumerate() {
            embeddings[[i, j]] = val;
        }
    }

    // Select optimal k
    let optimal_k = if n_faces <= 2 {
        n_faces
    } else {
        select_optimal_k(&embeddings, max_k)?
    };

    log_main!("video_analysis", "Selected optimal clusters", k = optimal_k);

    // Perform clustering
    let centroids = if optimal_k == 1 {
        // Single cluster - use mean as centroid
        let mean = embeddings.mean_axis(ndarray::Axis(0)).unwrap();
        let mut cent = Array2::zeros((1, emb_dim));
        cent.row_mut(0).assign(&mean);
        cent
    } else {
        kmeans_cluster(&embeddings, optimal_k, 100)?
    };

    // Assign cluster IDs to face records
    for record in face_records.iter_mut() {
        let emb_array = Array1::from_vec(record.embedding.clone());
        let (cluster_id, _) = find_nearest_centroid(&centroids, &emb_array)?;
        record.cluster_id = Some(cluster_id);
    }

    log_main!(
        "video_analysis",
        "Clustering complete",
        clusters = optimal_k
    );

    Ok(centroids)
}

/// Build cluster information for interactive mapping
///
/// # Arguments
/// * `face_records` - Face records with cluster assignments
/// * `centroids` - Cluster centroids
///
/// # Returns
/// Vector of cluster info sorted by frequency (most common first)
pub fn build_cluster_info(
    face_records: &[FaceRecord],
    centroids: &Array2<f32>,
) -> Result<Vec<ClusterInfo>> {
    let (n_clusters, _emb_dim) = centroids.dim();

    // Count faces per cluster and collect example faces
    let mut cluster_data: HashMap<usize, (usize, Vec<&FaceRecord>)> = HashMap::new();

    for record in face_records {
        if let Some(cluster_id) = record.cluster_id {
            let entry = cluster_data.entry(cluster_id).or_insert((0, Vec::new()));
            entry.0 += 1;
            entry.1.push(record);
        }
    }

    // Build cluster info
    let mut cluster_infos = Vec::new();

    for cluster_id in 0..n_clusters {
        if let Some((count, faces)) = cluster_data.get(&cluster_id) {
            // Select best example face (highest detection score)
            let best_face = faces
                .iter()
                .max_by(|a, b| {
                    a.face
                        .det_score
                        .partial_cmp(&b.face.det_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .ok_or_else(|| {
                    FaceSwapError::ProcessingError(format!("No faces in cluster {}", cluster_id))
                })?;

            cluster_infos.push(ClusterInfo {
                cluster_id,
                centroid: centroids.row(cluster_id).to_vec(),
                example_face: best_face.face.clone(),
                frame_count: *count,
                example_frame_idx: best_face.frame_idx,
            });
        }
    }

    // Sort by frequency (most common first)
    cluster_infos.sort_by(|a, b| b.frame_count.cmp(&a.frame_count));

    Ok(cluster_infos)
}

/// Build lookup table for per-frame face access during video processing
///
/// Maps (frame_idx, cluster_id) to all detected faces matching that cluster
/// in that frame. Used to quickly find which faces to swap on each frame.
///
/// # Arguments
/// * `face_records` - Face records with cluster assignments
///
/// # Returns
/// HashMap keyed by (frame_idx, cluster_id) with vectors of detected faces
pub fn build_face_lookup(
    face_records: &[FaceRecord],
) -> HashMap<(usize, usize), Vec<DetectedFace>> {
    let mut lookup: HashMap<(usize, usize), Vec<DetectedFace>> = HashMap::new();

    for record in face_records {
        if let Some(cluster_id) = record.cluster_id {
            let key = (record.frame_idx, cluster_id);
            lookup
                .entry(key)
                .or_insert_with(Vec::new)
                .push(record.face.clone());
        }
    }

    lookup
}

/// Save face records to a JSON file
pub fn save_face_records(face_records: &[FaceRecord], path: &str) -> Result<()> {
    let json = serde_json::to_vec(face_records).map_err(|e| {
        FaceSwapError::ProcessingError(format!("Failed to serialize face records: {}", e))
    })?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load face records from a JSON file
pub fn load_face_records(path: &str) -> Result<Vec<FaceRecord>> {
    let data = std::fs::read(path)?;
    let records: Vec<FaceRecord> = serde_json::from_slice(&data).map_err(|e| {
        FaceSwapError::ProcessingError(format!("Failed to deserialize face records: {}", e))
    })?;
    Ok(records)
}
