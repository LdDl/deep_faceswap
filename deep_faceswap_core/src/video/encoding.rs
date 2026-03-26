//! Video encoding from frames using ffmpeg

use crate::log_main;
use crate::types::{FaceSwapError, Result};

/// Encode frames into video file
///
/// # Arguments
/// * `frames_dir` - Directory containing frame files (frame_000000.png, ....)
/// * `output_path` - Path for output video file
/// * `original_video_path` - Path to original video (for audio and metadata)
///
/// # Returns
/// Ok(()) if encoding succeeds
pub fn encode_video(frames_dir: &str, output_path: &str, original_video_path: &str) -> Result<()> {
    // Get FPS and audio stream info from original video
    let (fps, has_audio) = get_video_metadata(original_video_path)?;

    log_main!(
        "video_encoding",
        "Starting video encoding",
        fps = fps,
        has_audio = has_audio
    );

    // Encode video from frames
    encode_frames_to_video(frames_dir, output_path, fps)?;

    // Copy audio from original video if present
    if has_audio {
        copy_audio_stream(original_video_path, output_path)?;
    }

    log_main!("video_encoding", "Video encoding complete");

    Ok(())
}

/// Get video metadata (FPS and audio presence)
fn get_video_metadata(video_path: &str) -> Result<(f64, bool)> {
    ffmpeg_next::init().map_err(|e| {
        FaceSwapError::ProcessingError(format!("Failed to initialize ffmpeg: {}", e))
    })?;

    let input_ctx = ffmpeg_next::format::input(&video_path)
        .map_err(|e| FaceSwapError::ProcessingError(format!("Failed to open video: {}", e)))?;

    // Get video stream for FPS
    let video_stream = input_ctx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| FaceSwapError::ProcessingError("No video stream found".to_string()))?;

    let fps = video_stream.avg_frame_rate();
    let fps_value = fps.numerator() as f64 / fps.denominator() as f64;

    // Check for audio stream
    let has_audio = input_ctx
        .streams()
        .best(ffmpeg_next::media::Type::Audio)
        .is_some();

    Ok((fps_value, has_audio))
}

/// Encode frames to video using ffmpeg
fn encode_frames_to_video(frames_dir: &str, output_path: &str, fps: f64) -> Result<()> {
    // Build ffmpeg command for encoding
    let frame_pattern = format!("{}/frame_%06d.png", frames_dir);

    // Use std::process::Command to call ffmpeg CLI
    // This is simpler than using ffmpeg-next's encoder API for this use case
    // y - Overwrite output file
    // libx264 - H.264 codec
    // yuv420p - Pixel format for compatibility
    // medium - Encoding preset
    // crf 23 - Quality (0-51, lower is better)
    let output = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-framerate")
        .arg(fps.to_string())
        .arg("-i")
        .arg(&frame_pattern)
        .arg("-c:v")
        .arg("libx264")
        .arg("-pix_fmt")
        .arg("yuv420p")
        .arg("-preset")
        .arg("medium")
        .arg("-crf")
        .arg("23")
        .arg(output_path)
        .output()
        .map_err(|e| FaceSwapError::ProcessingError(format!("Failed to execute ffmpeg: {}", e)))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(FaceSwapError::ProcessingError(format!(
            "ffmpeg encoding failed: {}",
            stderr
        )));
    }

    log_main!("video_encoding", "Frames encoded to video");

    Ok(())
}

/// Copy audio stream from original video to output video
fn copy_audio_stream(original_video_path: &str, output_video_path: &str) -> Result<()> {
    // Create temporary file for video with audio
    let temp_output = format!("{}.tmp.mp4", output_video_path);

    // Use ffmpeg to combine video (no audio) with audio from original
    // y - Overwrite output file
    // -c:v copy - Copy video stream without re-encoding
    // -c:a aac - Encode audio to AAC
    // -map 0:v:0 - Take video from first input
    // -map 1:a:0 - Take audio from second input
    // -shortest - Match shortest stream duration to avoid extra silence
    let output = std::process::Command::new("ffmpeg")
        .arg("-y")
        .arg("-i")
        // Video without audio
        .arg(output_video_path)
        .arg("-i")
        // Original video with audio
        .arg(original_video_path)
        .arg("-c:v")
        .arg("copy")
        .arg("-c:a")
        .arg("aac")
        .arg("-map")
        .arg("0:v:0")
        .arg("-map")
        .arg("1:a:0")
        .arg("-shortest")
        .arg(&temp_output)
        .output()
        .map_err(|e| {
            FaceSwapError::ProcessingError(format!("Failed to execute ffmpeg for audio: {}", e))
        })?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(FaceSwapError::ProcessingError(format!(
            "ffmpeg audio copy failed: {}",
            stderr
        )));
    }

    // Replace original output with temp file
    std::fs::rename(&temp_output, output_video_path)?;

    log_main!("video_encoding", "Audio stream copied");

    Ok(())
}
