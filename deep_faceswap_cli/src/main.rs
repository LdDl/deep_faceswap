use clap::Parser;
use deep_faceswap_core::occlusion::{FaceOccluder, ForegroundProtection, OcclusionOptions};
use deep_faceswap_core::utils::media::{is_video_file, validate_output_path};
use deep_faceswap_core::verbose::{set_verbose_level, VerboseLevel};
use deep_faceswap_core::Result;

#[derive(Parser)]
#[command(name = "deep-faceswap")]
#[command(version = "0.1.0")]
#[command(about = "Face swapping CLI tool", long_about = None)]
struct Cli {
    /// Verbose level: 0 (errors only), 1 (main events, default), 2 (additional details), 3 (all including debug)
    #[arg(short, long, global = true, default_value = "1")]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Parser)]
enum Commands {
    /// Swap faces in images or videos (auto-detected by file extension)
    Swap {
        /// Path to source image(s) - single path or comma-separated list (e.g., img1.jpg,img2.jpg)
        #[arg(short, long)]
        source: String,

        /// Path to target image or video (face to replace)
        #[arg(short, long)]
        target: String,

        /// Path to output image or video
        #[arg(short, long)]
        output: String,

        /// Path to detection model
        #[arg(long, default_value = "models/buffalo_l/det_10g.onnx")]
        detector: String,

        /// Path to recognition model
        #[arg(long, default_value = "models/buffalo_l/w600k_r50.onnx")]
        recognizer: String,

        /// Path to swapper model
        #[arg(long, default_value = "models/inswapper_128.onnx")]
        swapper: String,

        /// Enable face enhancement with GFPGAN
        #[arg(long)]
        enhance: bool,

        /// Path to enhancement model
        #[arg(long, default_value = "models/GFPGANv1.4.onnx")]
        enhancer: String,

        /// Enable mouth mask to preserve target's mouth expression
        #[arg(long)]
        mouth_mask: bool,

        /// Path to 106-point landmark model
        #[arg(long, default_value = "models/buffalo_l/2d106det.onnx")]
        landmark_model: String,

        /// Preserve objects covering the face, including hands and food
        #[arg(long)]
        preserve_foreground: bool,

        /// Path to XSeg ONNX model used by foreground preservation
        #[arg(long, default_value = "models/xseg_3.onnx")]
        occluder_model: String,

        /// Expand protected objects in aligned face pixels (0 to 16)
        #[arg(long, default_value = "2", value_parser = clap::value_parser!(u8).range(0..=16))]
        foreground_margin: u8,

        /// Soften protected edges in aligned face pixels (0 to 16)
        #[arg(long, default_value = "3", value_parser = clap::value_parser!(u8).range(0..=16))]
        foreground_blur: u8,

        /// Process multiple faces with interactive mapping
        #[arg(long)]
        multi_face: bool,

        /// Custom directory for temporary video frames (default: ./tmp/)
        #[arg(long)]
        video_tmp_dir: Option<String>,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    let verbose_level = match cli.verbose {
        0 => VerboseLevel::None,
        1 => VerboseLevel::Main,
        2 => VerboseLevel::Additional,
        3 => VerboseLevel::All,
        _ => {
            eprintln!("Invalid verbose level: {}. Using Main (1).", cli.verbose);
            VerboseLevel::Main
        }
    };

    set_verbose_level(verbose_level);

    match cli.command {
        Commands::Swap {
            source,
            target,
            output,
            detector,
            recognizer,
            swapper,
            enhance,
            enhancer,
            mouth_mask,
            landmark_model,
            preserve_foreground,
            occluder_model,
            foreground_margin,
            foreground_blur,
            multi_face,
            video_tmp_dir,
        } => {
            validate_output_path(&output, &target)?;

            let enhancer_model = if enhance {
                Some(enhancer.as_str())
            } else {
                None
            };
            let landmark_model = if mouth_mask {
                Some(landmark_model.as_str())
            } else {
                None
            };

            let mut occluder = if preserve_foreground {
                Some(FaceOccluder::new(&occluder_model)?)
            } else {
                None
            };
            let mut foreground = occluder.as_mut().map(|occluder| ForegroundProtection {
                occluder,
                options: OcclusionOptions {
                    margin: foreground_margin as usize,
                    blur: foreground_blur as usize,
                },
            });

            if is_video_file(&target) {
                deep_faceswap_core::swap_video_with_foreground(
                    &source,
                    &target,
                    &output,
                    &detector,
                    &recognizer,
                    &swapper,
                    enhancer_model,
                    landmark_model,
                    mouth_mask,
                    multi_face,
                    video_tmp_dir.as_deref(),
                    foreground.as_mut(),
                )?;
            } else {
                deep_faceswap_core::swap_faces_with_foreground(
                    &source,
                    &target,
                    &output,
                    &detector,
                    &recognizer,
                    &swapper,
                    enhancer_model,
                    landmark_model,
                    mouth_mask,
                    multi_face,
                    foreground.as_mut(),
                )?;
            }
        }
    }

    Ok(())
}
