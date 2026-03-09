//! Video frame extraction using ffmpeg

use crate::log_main;
use crate::types::{FaceSwapError, Result};
use std::fs;
use std::path::Path;

/// Extract frames from video file to output directory
///
/// # Arguments
/// * `video_path` - Path to input video file
/// * `output_dir` - Directory where frames will be saved
///
/// # Returns
/// Vector of paths to extracted frame files (sorted by frame number)
pub fn extract_frames(video_path: &str, output_dir: &str) -> Result<Vec<String>> {
    if !Path::new(video_path).exists() {
        return Err(FaceSwapError::InvalidInput(format!(
            "Video file not found: {}",
            video_path
        )));
    }

    fs::create_dir_all(output_dir)?;

    ffmpeg_next::init().map_err(|e| {
        FaceSwapError::ProcessingError(format!("Failed to initialize ffmpeg: {}", e))
    })?;

    let mut input_ctx = ffmpeg_next::format::input(&video_path)
        .map_err(|e| FaceSwapError::ProcessingError(format!("Failed to open video: {}", e)))?;

    let video_stream_index = input_ctx
        .streams()
        .best(ffmpeg_next::media::Type::Video)
        .ok_or_else(|| FaceSwapError::ProcessingError("No video stream found".to_string()))?
        .index();

    let video_stream = input_ctx.stream(video_stream_index).unwrap();

    let decoder_context =
        ffmpeg_next::codec::context::Context::from_parameters(video_stream.parameters()).map_err(
            |e| FaceSwapError::ProcessingError(format!("Failed to create decoder context: {}", e)),
        )?;

    let mut decoder = decoder_context.decoder().video().map_err(|e| {
        FaceSwapError::ProcessingError(format!("Failed to create video decoder: {}", e))
    })?;

    let mut frame_count = 0;
    let mut saved_frames: Vec<String> = Vec::new();

    // Process packets
    for (stream, packet) in input_ctx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet).map_err(|e| {
                FaceSwapError::ProcessingError(format!("Failed to send packet: {}", e))
            })?;

            let mut decoded_frame = ffmpeg_next::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                // Save frame as JPEG
                let frame_path = format!("{}/frame_{:06}.jpg", output_dir, frame_count);
                save_frame(&decoded_frame, &frame_path)?;
                saved_frames.push(frame_path);

                frame_count += 1;

                // Log every 30 frames
                if frame_count % 30 == 0 {
                    log_main!(
                        "video_extraction",
                        "Extracting frames",
                        frames_extracted = frame_count
                    );
                }
            }
        }
    }

    // Flush decoder
    decoder
        .send_eof()
        .map_err(|e| FaceSwapError::ProcessingError(format!("Failed to flush decoder: {}", e)))?;

    let mut decoded_frame = ffmpeg_next::util::frame::video::Video::empty();
    while decoder.receive_frame(&mut decoded_frame).is_ok() {
        let frame_path = format!("{}/frame_{:06}.jpg", output_dir, frame_count);
        save_frame(&decoded_frame, &frame_path)?;
        saved_frames.push(frame_path);
        frame_count += 1;
    }

    log_main!(
        "video_extraction",
        "Frame extraction complete",
        total_frames = frame_count
    );

    Ok(saved_frames)
}

/// Save video frame as JPEG image
fn save_frame(frame: &ffmpeg_next::util::frame::video::Video, output_path: &str) -> Result<()> {
    // Convert frame to RGB24 format
    let mut scaler = ffmpeg_next::software::scaling::context::Context::get(
        frame.format(),
        frame.width(),
        frame.height(),
        ffmpeg_next::format::Pixel::RGB24,
        frame.width(),
        frame.height(),
        ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
    )
    .map_err(|e| FaceSwapError::ProcessingError(format!("Failed to create scaler: {}", e)))?;

    let mut rgb_frame = ffmpeg_next::util::frame::video::Video::empty();
    scaler
        .run(frame, &mut rgb_frame)
        .map_err(|e| FaceSwapError::ProcessingError(format!("Failed to scale frame: {}", e)))?;

    let width = rgb_frame.width();
    let height = rgb_frame.height();
    let data = rgb_frame.data(0);

    // Create image buffer from frame data
    let img_buffer = image::RgbImage::from_fn(width, height, |x, y| {
        let offset = (y * rgb_frame.stride(0) as u32 + x * 3) as usize;
        image::Rgb([data[offset], data[offset + 1], data[offset + 2]])
    });

    // Save as JPEG
    img_buffer
        .save(output_path)
        .map_err(|e| FaceSwapError::Image(e))?;

    Ok(())
}
