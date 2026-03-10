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

    let estimated_frames = {
        let duration = video_stream.duration();
        let time_base = video_stream.time_base();
        let fps = video_stream.avg_frame_rate();
        let fps_value = fps.numerator() as f64 / fps.denominator() as f64;

        if duration > 0 && fps_value > 0.0 {
            let duration_secs =
                duration as f64 * time_base.numerator() as f64 / time_base.denominator() as f64;
            (duration_secs * fps_value) as usize
        } else {
            0
        }
    };

    let log_interval = if estimated_frames > 0 {
        std::cmp::max(10, estimated_frames / 10)
    } else {
        100
    };

    let mut decoder_context =
        ffmpeg_next::codec::context::Context::from_parameters(video_stream.parameters()).map_err(
            |e| FaceSwapError::ProcessingError(format!("Failed to create decoder context: {}", e)),
        )?;

    // Enable multi-threaded decoding (auto-detect thread count)
    decoder_context.set_threading(ffmpeg_next::codec::threading::Config {
        kind: ffmpeg_next::codec::threading::Type::Frame,
        count: 0,
    });

    let mut decoder = decoder_context.decoder().video().map_err(|e| {
        FaceSwapError::ProcessingError(format!("Failed to create video decoder: {}", e))
    })?;

    let mut frame_count = 0;
    let mut saved_frames: Vec<String> = Vec::new();

    // Scaler and MJPEG encoder initialized on first frame
    let mut scaler: Option<ffmpeg_next::software::scaling::context::Context> = None;
    let mut jpeg_encoder: Option<ffmpeg_next::encoder::Video> = None;
    let mut yuvj_frame = ffmpeg_next::util::frame::video::Video::empty();

    // Process packets
    for (stream, packet) in input_ctx.packets() {
        if stream.index() == video_stream_index {
            decoder.send_packet(&packet).map_err(|e| {
                FaceSwapError::ProcessingError(format!("Failed to send packet: {}", e))
            })?;

            let mut decoded_frame = ffmpeg_next::util::frame::video::Video::empty();
            while decoder.receive_frame(&mut decoded_frame).is_ok() {
                // Initialize scaler and encoder on first frame
                if scaler.is_none() {
                    let w = decoded_frame.width();
                    let h = decoded_frame.height();

                    scaler = Some(
                        ffmpeg_next::software::scaling::context::Context::get(
                            decoded_frame.format(),
                            w,
                            h,
                            ffmpeg_next::format::Pixel::YUVJ420P,
                            w,
                            h,
                            ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
                        )
                        .map_err(|e| {
                            FaceSwapError::ProcessingError(format!(
                                "Failed to create scaler: {}",
                                e
                            ))
                        })?,
                    );

                    // Create MJPEG encoder
                    let mjpeg_codec = ffmpeg_next::encoder::find(ffmpeg_next::codec::Id::MJPEG)
                        .ok_or_else(|| {
                            FaceSwapError::ProcessingError("MJPEG encoder not found".to_string())
                        })?;
                    let enc_ctx = ffmpeg_next::codec::context::Context::new_with_codec(mjpeg_codec);
                    let mut enc_params = enc_ctx.encoder().video().map_err(|e| {
                        FaceSwapError::ProcessingError(format!(
                            "Failed to create MJPEG encoder: {}",
                            e
                        ))
                    })?;
                    enc_params.set_width(w);
                    enc_params.set_height(h);
                    enc_params.set_format(ffmpeg_next::format::Pixel::YUVJ420P);
                    enc_params.set_time_base(ffmpeg_next::Rational::new(1, 25));
                    enc_params.set_quality(2);

                    let encoder = enc_params.open().map_err(|e| {
                        FaceSwapError::ProcessingError(format!(
                            "Failed to open MJPEG encoder: {}",
                            e
                        ))
                    })?;
                    jpeg_encoder = Some(encoder);
                }

                let frame_path = format!("{}/frame_{:06}.jpg", output_dir, frame_count);
                encode_frame_as_jpeg(
                    &decoded_frame,
                    &frame_path,
                    scaler.as_mut().unwrap(),
                    jpeg_encoder.as_mut().unwrap(),
                    &mut yuvj_frame,
                )?;
                saved_frames.push(frame_path);

                frame_count += 1;

                if frame_count % log_interval == 0 {
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
        encode_frame_as_jpeg(
            &decoded_frame,
            &frame_path,
            scaler.as_mut().unwrap(),
            jpeg_encoder.as_mut().unwrap(),
            &mut yuvj_frame,
        )?;
        saved_frames.push(frame_path);
        frame_count += 1;
    }

    Ok(saved_frames)
}

/// Encode a video frame as JPEG using ffmpeg MJPEG encoder
fn encode_frame_as_jpeg(
    frame: &ffmpeg_next::util::frame::video::Video,
    output_path: &str,
    scaler: &mut ffmpeg_next::software::scaling::context::Context,
    encoder: &mut ffmpeg_next::encoder::Video,
    yuvj_frame: &mut ffmpeg_next::util::frame::video::Video,
) -> Result<()> {
    // Convert to YUVJ420P for MJPEG encoder
    scaler
        .run(frame, yuvj_frame)
        .map_err(|e| FaceSwapError::ProcessingError(format!("Failed to scale frame: {}", e)))?;

    // Encode frame
    encoder.send_frame(yuvj_frame).map_err(|e| {
        FaceSwapError::ProcessingError(format!("Failed to send frame to encoder: {}", e))
    })?;

    let mut packet = ffmpeg_next::Packet::empty();
    if encoder.receive_packet(&mut packet).is_ok() {
        if let Some(data) = packet.data() {
            std::fs::write(output_path, data)?;
        }
    }

    Ok(())
}
