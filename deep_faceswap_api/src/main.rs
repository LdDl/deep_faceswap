//! Deep FaceSwap REST API server

mod error;
mod jobs;
mod services;
mod state;

use actix_cors::Cors;
use actix_web::{error as actix_error, http, web, App, HttpResponse, HttpServer};
use clap::Parser;
use deep_faceswap_core::detection::FaceDetector;
use deep_faceswap_core::enhancer::FaceEnhancer;
use deep_faceswap_core::landmark::LandmarkDetector;
use deep_faceswap_core::recognition::FaceRecognizer;
use deep_faceswap_core::swapper::FaceSwapper;
use deep_faceswap_core::verbose::{set_verbose_level, VerboseLevel};
use deep_faceswap_core::log_main;

#[derive(Parser)]
#[command(name = "deep-faceswap-api")]
#[command(version = "0.1.0")]
#[command(about = "Face swapping REST API server")]
struct Cli {
    /// Verbose level: 0 (errors only), 1 (main events, default), 2 (additional details), 3 (all including debug)
    #[arg(short, long, default_value = "1")]
    verbose: u8,

    /// Host to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value = "36000")]
    port: u16,

    /// Path to detection model
    #[arg(long, default_value = "models/buffalo_l/det_10g.onnx")]
    detector: String,

    /// Path to recognition model
    #[arg(long, default_value = "models/buffalo_l/w600k_r50.onnx")]
    recognizer: String,

    /// Path to swapper model
    #[arg(long, default_value = "models/inswapper_128.onnx")]
    swapper: String,

    /// Path to enhancement model (enables GFPGAN)
    #[arg(long)]
    enhancer: Option<String>,

    /// Path to 106-point landmark model (enables mouth mask)
    #[arg(long)]
    landmark_model: Option<String>,

    /// Allowed directories for file browser (comma-separated, empty = allow all)
    #[arg(long, default_value = "")]
    allowed_dir: String,

    /// Base directory for temporary files (session data, video frames)
    #[arg(long, default_value = "./tmp/api_sessions")]
    tmp_dir: String,

    /// Directory with SvelteKit build for UI
    #[arg(long, default_value = "")]
    ui_dir: String,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let cli = Cli::parse();

    let verbose_level = match cli.verbose {
        0 => VerboseLevel::None,
        1 => VerboseLevel::Main,
        2 => VerboseLevel::Additional,
        3.. => VerboseLevel::All,
    };
    set_verbose_level(verbose_level);

    // Set UI_DIR for services.rs to read
    if !cli.ui_dir.is_empty() {
        std::env::set_var("UI_DIR", &cli.ui_dir);
    }

    log_main!("startup", "Loading face detector", path = &cli.detector);
    let detector = FaceDetector::new(&cli.detector)
        .unwrap_or_else(|e| panic!("Failed to load detector '{}': {}", cli.detector, e));

    log_main!("startup", "Loading face recognizer", path = &cli.recognizer);
    let recognizer = FaceRecognizer::new(&cli.recognizer)
        .unwrap_or_else(|e| panic!("Failed to load recognizer '{}': {}", cli.recognizer, e));

    log_main!("startup", "Loading face swapper", path = &cli.swapper);
    let swapper = FaceSwapper::new(&cli.swapper)
        .unwrap_or_else(|e| panic!("Failed to load swapper '{}': {}", cli.swapper, e));

    let enhancer = cli.enhancer.as_ref().map(|path| {
        log_main!("startup", "Loading face enhancer", path = path);
        FaceEnhancer::new(path)
            .unwrap_or_else(|e| panic!("Failed to load enhancer '{}': {}", path, e))
    });

    let landmark_detector = cli.landmark_model.as_ref().map(|path| {
        log_main!("startup", "Loading landmark detector", path = path);
        LandmarkDetector::new(path)
            .unwrap_or_else(|e| panic!("Failed to load landmark model '{}': {}", path, e))
    });

    let model_count = 3 + enhancer.as_ref().map_or(0, |_| 1) + landmark_detector.as_ref().map_or(0, |_| 1);
    log_main!("startup", "All models loaded", count = model_count);

    // Parse allowed directories
    let allowed_dirs: Vec<String> = if cli.allowed_dir.is_empty() {
        Vec::new()
    } else {
        cli.allowed_dir
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    };

    if !allowed_dirs.is_empty() {
        let dirs_str = allowed_dirs.join(", ");
        log_main!("startup", "File browser restricted", dirs = dirs_str.as_str());
    }

    if !cli.ui_dir.is_empty() {
        log_main!("startup", "Serving UI", path = &cli.ui_dir);
    }

    let app_state = web::Data::new(state::AppState::new(
        detector,
        recognizer,
        swapper,
        enhancer,
        landmark_detector,
        allowed_dirs,
        cli.tmp_dir,
    ));

    let bind_address = format!("{}:{}", cli.host, cli.port);
    log_main!("startup", "Starting server", address = bind_address.as_str());

    HttpServer::new(move || {
        let cors = Cors::default()
            .allow_any_origin()
            .allowed_headers(vec![
                http::header::ORIGIN,
                http::header::CONTENT_TYPE,
                http::header::CONTENT_LENGTH,
                http::header::ACCEPT,
                http::header::ACCEPT_ENCODING,
            ])
            .allowed_methods(vec!["GET", "POST"])
            .max_age(3600);

        App::new()
            .wrap(cors)
            .app_data(app_state.clone())
            .app_data(web::JsonConfig::default().error_handler(|err, _req| {
                actix_error::InternalError::from_response(
                    "",
                    HttpResponse::BadRequest()
                        .content_type("application/json")
                        .body(format!(r#"{{"error_text":"{}"}}"#, err)),
                )
                .into()
            }))
            .configure(services::init_routes)
    })
    .bind(&bind_address)
    .unwrap_or_else(|_| panic!("Could not bind server to {}", &bind_address))
    .run()
    .await
}
