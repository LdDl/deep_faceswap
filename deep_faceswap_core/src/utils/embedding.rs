//! Embedding similarity utilities
//!
//! Cosine similarity: https://en.wikipedia.org/wiki/Cosine_similarity

use crate::alignment;
use crate::recognition::FaceRecognizer;
use crate::types::{DetectedFace, Result};
use ndarray::Array3;

/// Calculate cosine similarity between two embeddings
///
/// # Arguments
/// * `emb1` - First embedding
/// * `emb2` - Second embedding
///
/// # Returns
/// Similarity score (0.0 to 1.0, higher is more similar)
///
/// # Example
/// ```
/// use deep_faceswap_core::utils::embedding::cosine_similarity;
///
/// let emb1 = vec![1.0, 0.0, 0.0];
/// let emb2 = vec![0.0, 1.0, 0.0];
/// assert!(cosine_similarity(&emb1, &emb2) < 0.01);
///
/// let emb3 = vec![1.0, 2.0, 3.0];
/// assert!((cosine_similarity(&emb3, &emb3) - 1.0).abs() < 0.001);
/// ```
pub fn cosine_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
    if emb1.len() != emb2.len() {
        return 0.0;
    }

    let dot_product: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();

    let norm1: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm2: f32 = emb2.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm1 == 0.0 || norm2 == 0.0 {
        return 0.0;
    }

    (dot_product / (norm1 * norm2)).max(0.0).min(1.0)
}

/// Find face most similar to reference embedding
///
/// # Arguments
/// * `faces` - Detected faces in current frame
/// * `reference_embedding` - Reference face embedding to match
/// * `image` - Frame image (for alignment)
/// * `recognizer` - Face recognizer
/// * `threshold` - Minimum similarity threshold (default 0.5)
///
/// # Returns
/// Most similar face if similarity > threshold, otherwise None
pub fn find_most_similar_face<'a>(
    faces: &'a [DetectedFace],
    reference_embedding: &[f32],
    image: &Array3<u8>,
    recognizer: &mut FaceRecognizer,
    threshold: f32,
) -> Result<Option<&'a DetectedFace>> {
    let mut best_face = None;
    let mut best_similarity = threshold;

    for face in faces {
        let aligned = alignment::align_face(image, face, 112)?;
        let embedding = recognizer.extract_embedding(&aligned.aligned_image)?;
        let similarity = cosine_similarity(&embedding, reference_embedding);

        if similarity > best_similarity {
            best_similarity = similarity;
            best_face = Some(face);
        }
    }

    Ok(best_face)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 0.001;

    #[test]
    fn test_cosine_similarity_identical() {
        let emb = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&emb, &emb);
        assert!((sim - 1.0).abs() < EPS);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![0.0, 1.0];
        let sim = cosine_similarity(&emb1, &emb2);
        assert!(sim.abs() < EPS);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let emb1 = vec![1.0, 0.0];
        let emb2 = vec![-1.0, 0.0];
        // Clamped to 0.0
        let sim = cosine_similarity(&emb1, &emb2);
        assert!(sim < EPS);
    }
}
