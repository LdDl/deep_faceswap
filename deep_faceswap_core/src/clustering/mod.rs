//! Face clustering using K-means algorithm

pub mod kmeans;

pub use kmeans::{find_nearest_centroid, kmeans_cluster, select_optimal_k};
