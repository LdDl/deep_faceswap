use clap::Parser;
use deep_faceswap_core::Result;

#[derive(Parser)]
#[command(name = "deep-faceswap")]
#[command(version = "0.1.0")]
#[command(about = "Face swapping CLI tool", long_about = None)]
struct Cli {
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
    },
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Swap { source, target, output } => {
            deep_faceswap_core::swap_faces(&source, &target, &output)?;
        }
    }

    Ok(())
}
