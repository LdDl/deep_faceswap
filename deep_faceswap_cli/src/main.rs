use clap::Parser;
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
    /// Swap faces between two images
    Swap {
        /// Path to source image (face to extract)
        #[arg(short, long)]
        source: String,

        /// Path to target image (face to replace)
        #[arg(short, long)]
        target: String,

        /// Path to output image
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
        } => {
            deep_faceswap_core::swap_faces(
                &source,
                &target,
                &output,
                &detector,
                &recognizer,
                &swapper,
            )?;
        }
    }

    Ok(())
}
