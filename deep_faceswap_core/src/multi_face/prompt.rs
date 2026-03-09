//! Interactive CLI prompts for face selection

use crate::types::{FaceCropInfo, FaceMapping, FaceSwapError, Result};
use std::io::{self, Write};

/// Print face crop information to console
fn print_face_crops(crops: &[FaceCropInfo], label: &str) {
    println!(
        "\n{} faces saved to: ./tmp/face_crops/{}/",
        label,
        label.to_lowercase()
    );
    for crop in crops {
        let filename_part = std::path::Path::new(&crop.crop_path)
            .file_stem()
            .and_then(|s| s.to_str())
            .and_then(|s| s.split("_face_").next())
            .unwrap_or("unknown");

        if label.to_lowercase() == "source" && filename_part != "face" {
            println!(
                "  [{}] {} - {} (score: {:.2})",
                crop.index, filename_part, crop.crop_path, crop.face.det_score
            );
        } else {
            println!(
                "  [{}] {} (score: {:.2})",
                crop.index, crop.crop_path, crop.face.det_score
            );
        }
    }
    println!();
}

/// Prompt user to select target faces (1:N case)
///
/// # Arguments
/// * `source_crops` - Source face crops (should be length 1)
/// * `target_crops` - Target face crops
///
/// # Returns
/// Vector of selected target indices
pub fn prompt_target_selection(
    source_crops: &[FaceCropInfo],
    target_crops: &[FaceCropInfo],
) -> Result<Vec<usize>> {
    print_face_crops(source_crops, "Source");
    print_face_crops(target_crops, "Target");

    println!(
        "One source face detected, {} target faces detected.",
        target_crops.len()
    );
    println!("Enter target face indices to swap (comma-separated), or 'all' for all targets:");
    print!("> ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        return Err(FaceSwapError::UserCancelled);
    }

    if input.eq_ignore_ascii_case("all") {
        return Ok((0..target_crops.len()).collect());
    }

    // Parse comma-separated indices
    let mut indices = Vec::new();
    for part in input.split(',') {
        let part = part.trim();
        match part.parse::<usize>() {
            Ok(idx) => {
                if idx >= target_crops.len() {
                    return Err(FaceSwapError::InvalidMapping(format!(
                        "Target index {} out of range (max {})",
                        idx,
                        target_crops.len() - 1
                    )));
                }
                indices.push(idx);
            }
            Err(_) => {
                return Err(FaceSwapError::InvalidMapping(format!(
                    "Invalid index: '{}'",
                    part
                )));
            }
        }
    }

    if indices.is_empty() {
        return Err(FaceSwapError::InvalidMapping(
            "No valid indices provided".to_string(),
        ));
    }

    Ok(indices)
}

/// Prompt user to select source face (N:1 case)
///
/// # Arguments
/// * `source_crops` - Source face crops
/// * `target_crops` - Target face crops (should be length 1)
///
/// # Returns
/// Selected source index
pub fn prompt_source_selection(
    source_crops: &[FaceCropInfo],
    target_crops: &[FaceCropInfo],
) -> Result<usize> {
    print_face_crops(source_crops, "Source");
    print_face_crops(target_crops, "Target");

    println!(
        "{} source faces detected, one target face detected.",
        source_crops.len()
    );
    println!("Enter source face index to use:");
    print!("> ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        return Err(FaceSwapError::UserCancelled);
    }

    match input.parse::<usize>() {
        Ok(idx) => {
            if idx >= source_crops.len() {
                return Err(FaceSwapError::InvalidMapping(format!(
                    "Source index {} out of range (max {})",
                    idx,
                    source_crops.len() - 1
                )));
            }
            Ok(idx)
        }
        Err(_) => Err(FaceSwapError::InvalidMapping(format!(
            "Invalid index: '{}'",
            input
        ))),
    }
}

/// Prompt user for full face mapping (N:N case)
///
/// # Arguments
/// * `source_crops` - Source face crops
/// * `target_crops` - Target face crops
///
/// # Returns
/// Vector of FaceMapping (source->target pairs)
pub fn prompt_full_mapping(
    source_crops: &[FaceCropInfo],
    target_crops: &[FaceCropInfo],
) -> Result<Vec<FaceMapping>> {
    print_face_crops(source_crops, "Source");
    print_face_crops(target_crops, "Target");

    println!(
        "{} source faces detected, {} target faces detected.",
        source_crops.len(),
        target_crops.len()
    );
    println!("Enter face mappings in format 'S:T,S:T' (e.g., '0:0,1:1'):");
    println!("Where S is source index and T is target index.");
    print!("> ");
    io::stdout().flush()?;

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    let input = input.trim();

    if input.is_empty() {
        return Err(FaceSwapError::UserCancelled);
    }

    let mut mappings = Vec::new();

    for part in input.split(',') {
        let part = part.trim();
        let tokens: Vec<&str> = part.split(':').collect();

        if tokens.len() != 2 {
            return Err(FaceSwapError::InvalidMapping(format!(
                "Invalid mapping format: '{}' (expected 'S:T')",
                part
            )));
        }

        let source_idx = tokens[0].trim().parse::<usize>().map_err(|_| {
            FaceSwapError::InvalidMapping(format!("Invalid source index: '{}'", tokens[0]))
        })?;

        let target_idx = tokens[1].trim().parse::<usize>().map_err(|_| {
            FaceSwapError::InvalidMapping(format!("Invalid target index: '{}'", tokens[1]))
        })?;

        if source_idx >= source_crops.len() {
            return Err(FaceSwapError::InvalidMapping(format!(
                "Source index {} out of range (max {})",
                source_idx,
                source_crops.len() - 1
            )));
        }

        if target_idx >= target_crops.len() {
            return Err(FaceSwapError::InvalidMapping(format!(
                "Target index {} out of range (max {})",
                target_idx,
                target_crops.len() - 1
            )));
        }

        mappings.push(FaceMapping {
            source_idx,
            target_idx,
        });
    }

    if mappings.is_empty() {
        return Err(FaceSwapError::InvalidMapping(
            "No valid mappings provided".to_string(),
        ));
    }

    Ok(mappings)
}
