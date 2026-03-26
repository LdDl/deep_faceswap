const VIDEO_EXTENSIONS: &[&str] = &[
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v", ".mpg", ".mpeg",
];

const IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif",
];

pub fn is_video_file(path: &str) -> bool {
    let path_lower = path.to_lowercase();
    VIDEO_EXTENSIONS.iter().any(|ext| path_lower.ends_with(ext))
}

pub fn is_image_file(path: &str) -> bool {
    let path_lower = path.to_lowercase();
    IMAGE_EXTENSIONS.iter().any(|ext| path_lower.ends_with(ext))
}

/// Check that output path has a valid extension matching the expected media type.
pub fn validate_output_path(output: &str, target: &str) -> crate::Result<()> {
    if is_video_file(target) && !is_video_file(output) {
        return Err(crate::FaceSwapError::InvalidInput(format!(
            "Output path '{}' must have a video extension (e.g. .mp4) when target is a video",
            output
        )));
    }

    if is_image_file(target) && !is_image_file(output) {
        return Err(crate::FaceSwapError::InvalidInput(format!(
            "Output path '{}' must have an image extension (e.g. .jpg, .png) when target is an image",
            output
        )));
    }

    Ok(())
}
