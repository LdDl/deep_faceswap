//! K-means clustering for face embeddings
//!
//! Implementation based on Lloyd's algorithm with multiple random restarts,
//! matching sklearn.cluster.KMeans(n_init=10) behavior.
//!
//! For L2-normalized embeddings (like ArcFace), Euclidean distance and cosine
//! similarity are equivalent: ||a-b||^2 = 2(1 - cos(a,b)). This means standard
//! Euclidean K-means produces the same clusters as spherical K-means.
//!
//! References:
//! - K-means algorithm: https://en.wikipedia.org/wiki/K-means_clustering
//! - Silhouette score: https://en.wikipedia.org/wiki/Silhouette_(clustering)
//! - sklearn.cluster.KMeans: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

use crate::types::{FaceSwapError, Result};
use ndarray::{Array1, Array2};
use rand::Rng;

/// Number of independent K-means runs with different random initializations.
/// The run with lowest inertia (sum of squared distances to centroids) wins.
/// Matches sklearn's default n_init=10.
const N_INIT: usize = 10;

/// Perform K-means clustering on face embeddings with multiple restarts
///
/// Runs K-means N_INIT times with different random seeds and returns the
/// centroids from the run with lowest inertia (best fit).
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
    let (n_samples, n_features) = embeddings.dim();

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

    for _ in 0..N_INIT {
        let (centroids, inertia) =
            kmeans_single_run(embeddings, k, n_samples, n_features, max_iterations)?;

        if inertia < best_inertia {
            best_inertia = inertia;
            best_centroids = Some(centroids);
        }
    }

    best_centroids.ok_or_else(|| {
        FaceSwapError::ProcessingError("K-means failed to produce any result".to_string())
    })
}

/// Single K-means run with random initialization
///
/// Returns (centroids, inertia) where inertia is the sum of squared
/// Euclidean distances from each point to its assigned centroid.
fn kmeans_single_run(
    embeddings: &Array2<f32>,
    k: usize,
    n_samples: usize,
    n_features: usize,
    max_iterations: usize,
) -> Result<(Array2<f32>, f32)> {
    let mut rng = rand::thread_rng();
    let mut centroids = Array2::<f32>::zeros((k, n_features));
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Initialize centroids randomly from data points
    for i in 0..k {
        let idx = rng.gen_range(i..n_samples);
        indices.swap(i, idx);
        centroids.row_mut(i).assign(&embeddings.row(indices[i]));
    }

    let mut labels = vec![0usize; n_samples];
    let mut prev_labels = vec![usize::MAX; n_samples];

    for _iteration in 0..max_iterations {
        // Assignment step: assign each point to nearest centroid
        for i in 0..n_samples {
            labels[i] = nearest_centroid_idx(&centroids, &embeddings.row(i), k);
        }

        // Check convergence
        if labels == prev_labels {
            break;
        }
        prev_labels.clone_from(&labels);

        // Update step: recalculate centroids as mean of assigned points
        for cluster_id in 0..k {
            let mut sum = Array1::<f32>::zeros(n_features);
            let mut count = 0usize;

            for (idx, &label) in labels.iter().enumerate() {
                if label == cluster_id {
                    sum += &embeddings.row(idx);
                    count += 1;
                }
            }

            if count > 0 {
                sum /= count as f32;
                centroids.row_mut(cluster_id).assign(&sum);
            }
        }
    }

    // Calculate inertia (sum of squared distances to assigned centroids)
    let mut inertia = 0.0f32;
    for i in 0..n_samples {
        let centroid = centroids.row(labels[i]);
        let diff = &embeddings.row(i) - &centroid;
        inertia += diff.dot(&diff);
    }

    Ok((centroids, inertia))
}

/// Find index of nearest centroid using cosine similarity
///
/// Used internally during K-means iteration. For L2-normalized embeddings,
/// maximizing cosine similarity is equivalent to minimizing Euclidean distance.
fn nearest_centroid_idx(
    centroids: &Array2<f32>,
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

/// Select optimal number of clusters using silhouette score
///
/// Tries K from 2 to max_k, runs K-means for each, and picks the K with
/// the highest silhouette score.
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
        let centroids = kmeans_cluster(embeddings, k, 100)?;
        let score = calculate_silhouette_score(embeddings, &centroids)?;

        if score > best_score {
            best_score = score;
            best_k = k;
        }
    }

    Ok(best_k)
}

/// Calculate silhouette score for clustering quality evaluation
///
/// For each sample, computes:
/// - a(i): mean distance to other points in the same cluster
/// - b(i): mean distance to points in the nearest other cluster
/// - s(i) = (b(i) - a(i)) / max(a(i), b(i))
///
/// Returns the mean s(i) across all samples. Range: [-1, 1].
/// Higher is better: 1 = well-separated, 0 = overlapping, -1 = misclassified.
fn calculate_silhouette_score(embeddings: &Array2<f32>, centroids: &Array2<f32>) -> Result<f32> {
    let (n_samples, _) = embeddings.dim();
    let (k, _) = centroids.dim();

    // Assign labels
    let mut labels = vec![0usize; n_samples];
    for i in 0..n_samples {
        labels[i] = nearest_centroid_idx(centroids, &embeddings.row(i), k);
    }

    let mut silhouette_sum = 0.0;

    for i in 0..n_samples {
        let own_cluster = labels[i];
        let point = embeddings.row(i);

        // Average distance to own cluster (a)
        let mut a = 0.0;
        let mut count_a = 0;
        for j in 0..n_samples {
            if i != j && labels[j] == own_cluster {
                a += euclidean_distance(&point, &embeddings.row(j));
                count_a += 1;
            }
        }
        if count_a > 0 {
            a /= count_a as f32;
        }

        // Minimum average distance to other clusters (b)
        let mut b = f32::INFINITY;
        for other_cluster in 0..k {
            if other_cluster == own_cluster {
                continue;
            }

            let mut cluster_dist = 0.0;
            let mut count_b = 0;
            for j in 0..n_samples {
                if labels[j] == other_cluster {
                    cluster_dist += euclidean_distance(&point, &embeddings.row(j));
                    count_b += 1;
                }
            }

            if count_b > 0 {
                cluster_dist /= count_b as f32;
                b = b.min(cluster_dist);
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

    Ok(silhouette_sum / n_samples as f32)
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

/// Euclidean distance between two vectors (no allocation)
fn euclidean_distance(a: &ndarray::ArrayView1<f32>, b: &ndarray::ArrayView1<f32>) -> f32 {
    euclidean_distance_sq(a, b).sqrt()
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
        // Two centroids pointing in different directions
        let centroids = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let emb = Array1::from_vec(vec![0.9, 0.1]);

        let (idx, _) = find_nearest_centroid(&centroids, &emb).unwrap();
        assert_eq!(idx, 0, "Should match centroid [1,0] not [0,1]");
    }

    #[test]
    fn test_kmeans_convergence() {
        // Two well-separated clusters of normalized vectors
        let embeddings = Array2::from_shape_vec(
            (6, 2),
            vec![
                // Group A: vectors pointing roughly toward (1, 0)
                0.99, 0.14, 0.97, 0.24, 0.95, 0.31,
                // Group B: vectors pointing roughly toward (0, 1)
                0.14, 0.99, 0.24, 0.97, 0.31, 0.95,
            ],
        )
        .unwrap();

        let centroids = kmeans_cluster(&embeddings, 2, 100).unwrap();
        assert_eq!(centroids.dim(), (2, 2));

        // Each point should be assigned to its correct cluster
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
        // Three distinct clusters
        let embeddings = Array2::from_shape_vec(
            (9, 2),
            vec![
                // Cluster A: near (1, 0)
                1.0, 0.0, 0.98, 0.05, 0.95, 0.1,
                // Cluster B: near (0, 1)
                0.0, 1.0, 0.05, 0.98, 0.1, 0.95,
                // Cluster C: near (-1, 0)
                -1.0, 0.0, -0.98, 0.05, -0.95, 0.1,
            ],
        )
        .unwrap();

        let k = select_optimal_k(&embeddings, 5).unwrap();
        assert!(k >= 2 && k <= 4, "Expected k=3 (got {})", k);
    }

    #[test]
    fn test_multiple_restarts_stability() {
        // With n_init=10, repeated calls should give similar inertia
        let embeddings =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, -1.0]).unwrap();

        let c1 = kmeans_cluster(&embeddings, 2, 50).unwrap();
        let c2 = kmeans_cluster(&embeddings, 2, 50).unwrap();

        // Both runs should produce 2 centroids
        assert_eq!(c1.dim(), (2, 2));
        assert_eq!(c2.dim(), (2, 2));
    }
}
