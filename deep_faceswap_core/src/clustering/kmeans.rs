//! K-means clustering for face embeddings
//!
//! Optimized implementation with:
//! - K-means++ initialization for faster convergence
//! - Sampled silhouette score for O(S^2) instead of O(N^2) model selection
//! - Single-pass centroid updates and pre-computed norms
//!
//! For L2-normalized embeddings (like ArcFace), Euclidean distance and cosine
//! similarity are equivalent: ||a-b||^2 = 2(1 - cos(a,b)). This means standard
//! Euclidean K-means produces the same clusters as spherical K-means.

use crate::types::{FaceSwapError, Result};
use ndarray::{Array1, Array2};
use rand::seq::SliceRandom;
use rand::Rng;

/// Number of independent K-means runs for the final clustering.
const N_INIT_FINAL: usize = 10;

/// Fewer restarts during K-selection (rough clustering is sufficient for silhouette comparison).
const N_INIT_SELECTION: usize = 3;

/// Maximum samples used for silhouette score estimation.
/// 2000 is more than enough for stable K selection with K <= 10.
const SILHOUETTE_SAMPLE_SIZE: usize = 2000;

/// Perform K-means clustering on face embeddings with multiple restarts
///
/// Runs K-means N_INIT_FINAL times with different random seeds and returns the
/// centroids from the run with lowest inertia (best fit).
/// Uses K-means++ initialization for faster convergence.
///
/// # Arguments
/// * `embeddings` - Face embeddings (N x D matrix, where N=faces, D=embedding_dim)
/// * `k` - Number of clusters
/// * `max_iterations` - Maximum iterations per run for convergence
///
/// # Returns
/// Centroids (K x D matrix) from the best run
///
/// # Example
/// ```
/// use deep_faceswap_core::clustering::kmeans_cluster;
/// use ndarray::Array2;
///
/// let embeddings = Array2::from_shape_vec(
///     (4, 2),
///     vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9],
/// ).unwrap();
///
/// let centroids = kmeans_cluster(&embeddings, 2, 100).unwrap();
/// assert_eq!(centroids.dim(), (2, 2));
/// ```
pub fn kmeans_cluster(
    embeddings: &Array2<f32>,
    k: usize,
    max_iterations: usize,
) -> Result<Array2<f32>> {
    kmeans_cluster_with_restarts(embeddings, k, max_iterations, N_INIT_FINAL)
}

fn kmeans_cluster_with_restarts(
    embeddings: &Array2<f32>,
    k: usize,
    max_iterations: usize,
    n_init: usize,
) -> Result<Array2<f32>> {
    let (n_samples, _) = embeddings.dim();

    if n_samples == 0 {
        return Err(FaceSwapError::ProcessingError(
            "Cannot cluster empty embeddings".to_string(),
        ));
    }

    if n_samples < k {
        return Err(FaceSwapError::ProcessingError(format!(
            "Cannot cluster {} samples into {} clusters",
            n_samples, k
        )));
    }

    let mut best_centroids = None;
    let mut best_inertia = f32::INFINITY;

    for _ in 0..n_init {
        let (centroids, inertia) = kmeans_single_run(embeddings, k, max_iterations)?;

        if inertia < best_inertia {
            best_inertia = inertia;
            best_centroids = Some(centroids);
        }
    }

    best_centroids.ok_or_else(|| {
        FaceSwapError::ProcessingError("K-means failed to produce any result".to_string())
    })
}

/// Single K-means run with K-means++ initialization
///
/// Returns (centroids, inertia) where inertia is the sum of squared
/// Euclidean distances from each point to its assigned centroid.
/// Uses pre-computed centroid norms and single-pass centroid updates
/// for optimal performance.
fn kmeans_single_run(
    embeddings: &Array2<f32>,
    k: usize,
    max_iterations: usize,
) -> Result<(Array2<f32>, f32)> {
    let (n_samples, n_features) = embeddings.dim();

    // K-means++ initialization
    let mut centroids = kmeans_pp_init(embeddings, k);

    // Pre-allocate buffers reused across iterations
    let mut labels = vec![0usize; n_samples];
    let mut prev_labels = vec![usize::MAX; n_samples];
    let mut centroid_norms = vec![0.0f32; k];
    let mut accum = Array2::<f32>::zeros((k, n_features));
    let mut counts = vec![0usize; k];

    for _iteration in 0..max_iterations {
        // Pre-compute centroid norms once per iteration
        for c in 0..k {
            centroid_norms[c] = centroids.row(c).dot(&centroids.row(c)).sqrt();
        }

        // Assignment step: assign each point to nearest centroid using pre-computed norms
        for i in 0..n_samples {
            labels[i] =
                nearest_centroid_idx_fast(&centroids, &centroid_norms, &embeddings.row(i), k);
        }

        // Check convergence
        if labels == prev_labels {
            break;
        }
        prev_labels.clone_from(&labels);

        // Single-pass centroid update
        accum.fill(0.0);
        counts.fill(0);

        for (idx, &label) in labels.iter().enumerate() {
            let row = embeddings.row(idx);
            let acc_row = accum.row_mut(label);
            // Manual element-wise add to avoid ndarray allocation
            let acc_slice = acc_row.into_slice().unwrap();
            let row_slice = row.as_slice().unwrap();
            for j in 0..n_features {
                acc_slice[j] += row_slice[j];
            }
            counts[label] += 1;
        }

        for c in 0..k {
            if counts[c] > 0 {
                let inv = 1.0 / counts[c] as f32;
                let mut row = centroids.row_mut(c);
                let acc = accum.row(c);
                for j in 0..n_features {
                    row[j] = acc[j] * inv;
                }
            }
        }
    }

    // Calculate inertia without allocating diff vectors
    let mut inertia = 0.0f32;
    for i in 0..n_samples {
        inertia += euclidean_distance_sq(&embeddings.row(i), &centroids.row(labels[i]));
    }

    Ok((centroids, inertia))
}

/// K-means++ initialization: pick centroids with D^2 weighting
///
/// First centroid is chosen uniformly at random. Each subsequent centroid
/// is chosen with probability proportional to squared distance to nearest
/// existing centroid. This gives O(log k)-competitive initialization.
///
/// Reference: Arthur & Vassilvitskii, "k-means++: The Advantages of Careful Seeding", 2007
fn kmeans_pp_init(embeddings: &Array2<f32>, k: usize) -> Array2<f32> {
    let (n_samples, n_features) = embeddings.dim();
    let mut rng = rand::thread_rng();
    let mut centroids = Array2::<f32>::zeros((k, n_features));

    // First centroid: random
    let first = rng.gen_range(0..n_samples);
    centroids.row_mut(0).assign(&embeddings.row(first));

    // Distance from each point to nearest chosen centroid
    let mut min_dist = vec![f32::INFINITY; n_samples];

    for c in 1..k {
        // Update min distances with the last added centroid
        let last_centroid = centroids.row(c - 1);
        for i in 0..n_samples {
            let d = euclidean_distance_sq(&embeddings.row(i), &last_centroid);
            if d < min_dist[i] {
                min_dist[i] = d;
            }
        }

        // Weighted random selection proportional to D^2
        let total: f32 = min_dist.iter().sum();
        if total == 0.0 {
            // All remaining points are on top of existing centroids
            centroids.row_mut(c).assign(&embeddings.row(rng.gen_range(0..n_samples)));
            continue;
        }

        let threshold = rng.gen::<f32>() * total;
        let mut cumsum = 0.0;
        let mut chosen = 0;
        for (i, &d) in min_dist.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.row_mut(c).assign(&embeddings.row(chosen));
    }

    centroids
}

/// Fast nearest centroid using pre-computed norms (cosine similarity)
fn nearest_centroid_idx_fast(
    centroids: &Array2<f32>,
    centroid_norms: &[f32],
    embedding: &ndarray::ArrayView1<f32>,
    k: usize,
) -> usize {
    let emb_norm = embedding.dot(embedding).sqrt();
    if emb_norm == 0.0 {
        return 0;
    }

    let mut best_idx = 0;
    let mut best_similarity = f32::NEG_INFINITY;

    for i in 0..k {
        let cn = centroid_norms[i];
        if cn == 0.0 {
            continue;
        }

        let similarity = embedding.dot(&centroids.row(i)) / (emb_norm * cn);

        if similarity > best_similarity {
            best_similarity = similarity;
            best_idx = i;
        }
    }

    best_idx
}

/// Find nearest centroid for a given embedding using cosine similarity
///
/// # Arguments
/// * `centroids` - Centroids matrix (K x D)
/// * `embedding` - Face embedding (D-dimensional)
///
/// # Returns
/// (cluster_id, similarity_score)
///
/// # Example
/// ```
/// use deep_faceswap_core::clustering::find_nearest_centroid;
/// use ndarray::{Array1, Array2};
///
/// let centroids = Array2::from_shape_vec(
///     (2, 2),
///     vec![1.0, 0.0, 0.0, 1.0],
/// ).unwrap();
/// let embedding = Array1::from_vec(vec![0.9, 0.1]);
///
/// let (idx, score) = find_nearest_centroid(&centroids, &embedding).unwrap();
/// assert_eq!(idx, 0);
/// assert!(score > 0.9);
/// ```
pub fn find_nearest_centroid(
    centroids: &Array2<f32>,
    embedding: &Array1<f32>,
) -> Result<(usize, f32)> {
    let (k, _) = centroids.dim();

    if k == 0 {
        return Err(FaceSwapError::ProcessingError(
            "No centroids provided".to_string(),
        ));
    }

    let emb_norm = embedding.dot(embedding).sqrt();
    if emb_norm == 0.0 {
        return Err(FaceSwapError::ProcessingError(
            "Zero embedding vector".to_string(),
        ));
    }

    let mut best_idx = 0;
    let mut best_similarity = f32::NEG_INFINITY;

    for i in 0..k {
        let centroid = centroids.row(i);
        let cent_norm = centroid.dot(&centroid).sqrt();
        if cent_norm == 0.0 {
            continue;
        }

        let similarity = embedding.dot(&centroid) / (emb_norm * cent_norm);

        if similarity > best_similarity {
            best_similarity = similarity;
            best_idx = i;
        }
    }

    Ok((best_idx, best_similarity))
}

/// Find nearest centroid for an embedding view using cosine similarity
///
/// Same as [`find_nearest_centroid`] but accepts an `ArrayView1` instead
/// of `Array1`, avoiding unnecessary allocation when working with
/// matrix row views.
///
/// # Arguments
/// * `centroids` - Centroids matrix (K x D)
/// * `embedding` - Face embedding view (D-dimensional)
///
/// # Returns
/// (cluster_id, similarity_score)
pub fn find_nearest_centroid_view(
    centroids: &Array2<f32>,
    embedding: &ndarray::ArrayView1<f32>,
) -> Result<(usize, f32)> {
    let (k, _) = centroids.dim();

    if k == 0 {
        return Err(FaceSwapError::ProcessingError(
            "No centroids provided".to_string(),
        ));
    }

    let emb_norm = embedding.dot(embedding).sqrt();
    if emb_norm == 0.0 {
        return Err(FaceSwapError::ProcessingError(
            "Zero embedding vector".to_string(),
        ));
    }

    let mut best_idx = 0;
    let mut best_similarity = f32::NEG_INFINITY;

    for i in 0..k {
        let centroid = centroids.row(i);
        let cent_norm = centroid.dot(&centroid).sqrt();
        if cent_norm == 0.0 {
            continue;
        }

        let similarity = embedding.dot(&centroid) / (emb_norm * cent_norm);

        if similarity > best_similarity {
            best_similarity = similarity;
            best_idx = i;
        }
    }

    Ok((best_idx, best_similarity))
}

/// Select optimal number of clusters using sampled silhouette score
///
/// Tries K from 2 to max_k, runs K-means for each, and picks the K with
/// the highest silhouette score. Uses fewer restarts (N_INIT_SELECTION=3)
/// and sampled silhouette (SILHOUETTE_SAMPLE_SIZE=2000) for fast model selection.
///
/// # Arguments
/// * `embeddings` - Face embeddings (N x D)
/// * `max_k` - Maximum number of clusters to try
///
/// # Returns
/// Optimal k value
///
/// # Example
/// ```
/// use deep_faceswap_core::clustering::select_optimal_k;
/// use ndarray::Array2;
///
/// let embeddings = Array2::from_shape_vec(
///     (6, 2),
///     vec![
///         1.0, 0.0, 0.95, 0.1,
///         0.0, 1.0, 0.1, 0.95,
///         -1.0, 0.0, -0.95, 0.1,
///     ],
/// ).unwrap();
///
/// let k = select_optimal_k(&embeddings, 5).unwrap();
/// assert!(k >= 2);
/// ```
pub fn select_optimal_k(embeddings: &Array2<f32>, max_k: usize) -> Result<usize> {
    let (n_samples, _) = embeddings.dim();

    if n_samples <= 2 {
        return Ok(n_samples);
    }

    let max_k = max_k.min(n_samples);
    let mut best_k = 1;
    let mut best_score = f32::NEG_INFINITY;

    for k in 2..=max_k {
        let centroids =
            kmeans_cluster_with_restarts(embeddings, k, 100, N_INIT_SELECTION)?;
        let score = calculate_silhouette_score_sampled(embeddings, &centroids)?;

        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    Ok(best_k)
}

/// Sampled silhouette score for clustering quality evaluation
///
/// For each sample point, computes:
/// - a(i): mean distance to other points in the same cluster
/// - b(i): mean distance to points in the nearest other cluster
/// - s(i) = (b(i) - a(i)) / max(a(i), b(i))
///
/// Returns the mean s(i) across all sampled points. Range: [-1, 1].
/// Higher is better: 1 = well-separated, 0 = overlapping, -1 = misclassified.
///
/// When N > SILHOUETTE_SAMPLE_SIZE, randomly subsamples to keep runtime
/// at O(S^2 · D) instead of O(N^2 · D). This matches sklearn's
/// `silhouette_score(sample_size=...)` approach.
fn calculate_silhouette_score_sampled(
    embeddings: &Array2<f32>,
    centroids: &Array2<f32>,
) -> Result<f32> {
    let (n_samples, n_features) = embeddings.dim();
    let (k, _) = centroids.dim();

    // Subsample if dataset is large
    let sample_size = n_samples.min(SILHOUETTE_SAMPLE_SIZE);

    let (sample_embeddings, sample_labels) = if sample_size < n_samples {
        // Random sample without replacement
        let mut rng = rand::thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);
        indices.truncate(sample_size);

        // Build subsampled embeddings matrix
        let mut sub = Array2::<f32>::zeros((sample_size, n_features));
        for (row_idx, &idx) in indices.iter().enumerate() {
            let row_view = embeddings.row(idx);
            let src = row_view.as_slice().unwrap();
            let dst = sub.row_mut(row_idx).into_slice().unwrap();
            dst.copy_from_slice(src);
        }

        // Assign labels
        let mut centroid_norms = vec![0.0f32; k];
        for c in 0..k {
            centroid_norms[c] = centroids.row(c).dot(&centroids.row(c)).sqrt();
        }
        let labels: Vec<usize> = (0..sample_size)
            .map(|i| nearest_centroid_idx_fast(centroids, &centroid_norms, &sub.row(i), k))
            .collect();

        (sub, labels)
    } else {
        // Use all samples
        let mut centroid_norms = vec![0.0f32; k];
        for c in 0..k {
            centroid_norms[c] = centroids.row(c).dot(&centroids.row(c)).sqrt();
        }
        let labels: Vec<usize> = (0..n_samples)
            .map(|i| {
                nearest_centroid_idx_fast(centroids, &centroid_norms, &embeddings.row(i), k)
            })
            .collect();

        // Clone the embeddings view — needed for uniform return type
        (embeddings.to_owned(), labels)
    };

    let n = sample_labels.len();

    // Pre-compute cluster membership lists for O(1) iteration per cluster
    let mut cluster_members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &label) in sample_labels.iter().enumerate() {
        if label < k {
            cluster_members[label].push(i);
        }
    }

    let mut silhouette_sum = 0.0f64; // f64 for accumulation stability

    for i in 0..n {
        let own_cluster = sample_labels[i];
        let point = sample_embeddings.row(i);

        // a(i): mean distance to own cluster members
        let own_members = &cluster_members[own_cluster];
        let mut a = 0.0f64;
        let count_a = own_members.len() - 1; // exclude self
        if count_a > 0 {
            for &j in own_members {
                if j != i {
                    a += euclidean_distance_sq(&point, &sample_embeddings.row(j)).sqrt() as f64;
                }
            }
            a /= count_a as f64;
        }

        // b(i): min average distance to other clusters
        let mut b = f64::INFINITY;
        for other_cluster in 0..k {
            if other_cluster == own_cluster {
                continue;
            }
            let members = &cluster_members[other_cluster];
            if members.is_empty() {
                continue;
            }
            let mut cluster_dist = 0.0f64;
            for &j in members {
                cluster_dist +=
                    euclidean_distance_sq(&point, &sample_embeddings.row(j)).sqrt() as f64;
            }
            cluster_dist /= members.len() as f64;
            if cluster_dist < b {
                b = cluster_dist;
            }
        }

        let s_i = if a < b {
            1.0 - a / b
        } else if a > b {
            b / a - 1.0
        } else {
            0.0
        };

        silhouette_sum += s_i;
    }

    Ok((silhouette_sum / n as f64) as f32)
}

/// Squared Euclidean distance between two vectors (no sqrt, no allocation)
fn euclidean_distance_sq(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_simple() {
        let embeddings =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.1, 1.0, 5.0, 5.0, 5.1, 5.0]).unwrap();

        let centroids = kmeans_cluster(&embeddings, 2, 10).unwrap();
        assert_eq!(centroids.dim(), (2, 2));
    }

    #[test]
    fn test_find_nearest() {
        let centroids = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let emb = Array1::from_vec(vec![0.9, 0.1]);

        let (idx, _) = find_nearest_centroid(&centroids, &emb).unwrap();
        assert_eq!(idx, 0, "Should match centroid [1,0] not [0,1]");
    }

    #[test]
    fn test_kmeans_convergence() {
        let embeddings = Array2::from_shape_vec(
            (6, 2),
            vec![
                0.99, 0.14, 0.97, 0.24, 0.95, 0.31, 0.14, 0.99, 0.24, 0.97, 0.31, 0.95,
            ],
        )
        .unwrap();

        let centroids = kmeans_cluster(&embeddings, 2, 100).unwrap();
        assert_eq!(centroids.dim(), (2, 2));

        for i in 0..3 {
            let (idx_a, _) =
                find_nearest_centroid(&centroids, &embeddings.row(i).to_owned()).unwrap();
            let (idx_b, _) =
                find_nearest_centroid(&centroids, &embeddings.row(i + 3).to_owned()).unwrap();
            assert_ne!(
                idx_a, idx_b,
                "Points from different groups should be in different clusters"
            );
        }
    }

    #[test]
    fn test_select_optimal_k() {
        let embeddings = Array2::from_shape_vec(
            (9, 2),
            vec![
                1.0, 0.0, 0.98, 0.05, 0.95, 0.1, 0.0, 1.0, 0.05, 0.98, 0.1, 0.95, -1.0, 0.0,
                -0.98, 0.05, -0.95, 0.1,
            ],
        )
        .unwrap();

        let k = select_optimal_k(&embeddings, 5).unwrap();
        assert!(k >= 2 && k <= 4, "Expected k=3 (got {})", k);
    }

    #[test]
    fn test_multiple_restarts_stability() {
        let embeddings =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0])
                .unwrap();

        let c1 = kmeans_cluster(&embeddings, 2, 50).unwrap();
        let c2 = kmeans_cluster(&embeddings, 2, 50).unwrap();

        assert_eq!(c1.dim(), (2, 2));
        assert_eq!(c2.dim(), (2, 2));
    }

    #[test]
    fn test_kmeans_pp_init() {
        let embeddings = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9, -1.0, 0.0, -0.9, -0.1],
        )
        .unwrap();

        let centroids = kmeans_pp_init(&embeddings, 3);
        assert_eq!(centroids.dim(), (3, 2));

        // All centroids should be non-zero (selected from data)
        for c in 0..3 {
            let norm = centroids.row(c).dot(&centroids.row(c));
            assert!(norm > 0.0, "Centroid {} should be non-zero", c);
        }
    }

    #[test]
    fn test_sampled_silhouette_small_dataset() {
        // For small datasets (< SILHOUETTE_SAMPLE_SIZE), sampled = exact
        let embeddings = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 0.0, 0.9, 0.1, 0.0, 1.0, 0.1, 0.9, -1.0, 0.0, -0.9, -0.1],
        )
        .unwrap();

        let centroids = kmeans_cluster(&embeddings, 2, 100).unwrap();
        let score = calculate_silhouette_score_sampled(&embeddings, &centroids).unwrap();

        // Well-separated clusters should have positive silhouette
        assert!(score > 0.0, "Expected positive silhouette, got {}", score);
    }
}
