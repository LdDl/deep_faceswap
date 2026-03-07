use std::fmt;
use std::sync::OnceLock;
use tracing::{debug, info, trace, Level};
use tracing_subscriber::{
    fmt as tracing_fmt, layer::SubscriberExt, reload, util::SubscriberInitExt, EnvFilter,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(u8)]
pub enum VerboseLevel {
    None = 0,
    Main = 1,
    Additional = 2,
    All = 3,
}

impl fmt::Display for VerboseLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            VerboseLevel::None => "none",
            VerboseLevel::Main => "main",
            VerboseLevel::Additional => "additional",
            VerboseLevel::All => "all",
        };
        write!(f, "{}", s)
    }
}

impl From<VerboseLevel> for Level {
    fn from(level: VerboseLevel) -> Self {
        match level {
            VerboseLevel::None => Level::ERROR,
            VerboseLevel::Main => Level::INFO,
            VerboseLevel::Additional => Level::DEBUG,
            VerboseLevel::All => Level::TRACE,
        }
    }
}

impl From<VerboseLevel> for String {
    fn from(level: VerboseLevel) -> Self {
        match level {
            VerboseLevel::None => "error".to_string(),
            VerboseLevel::Main => "info".to_string(),
            VerboseLevel::Additional => "debug".to_string(),
            VerboseLevel::All => "trace".to_string(),
        }
    }
}

pub const EVENT_LOAD_MODEL: &str = "load_model";
pub const EVENT_LOAD_IMAGE: &str = "load_image";
pub const EVENT_DETECT_FACES: &str = "detect_faces";
pub const EVENT_FACE_DETECTED: &str = "face_detected";
pub const EVENT_EXTRACT_EMBEDDING: &str = "extract_embedding";
pub const EVENT_ALIGN_FACE: &str = "align_face";
pub const EVENT_SWAP_FACE: &str = "swap_face";
pub const EVENT_PASTE_BACK: &str = "paste_back";
pub const EVENT_SAVE_IMAGE: &str = "save_image";
pub const EVENT_COMPLETE: &str = "complete";

static VERBOSE_LEVEL: OnceLock<VerboseLevel> = OnceLock::new();
static LOGGER_INITIALIZED: OnceLock<bool> = OnceLock::new();
static RELOAD_HANDLE: OnceLock<reload::Handle<EnvFilter, tracing_subscriber::Registry>> =
    OnceLock::new();

pub fn init_logger() {
    if LOGGER_INITIALIZED.set(true).is_ok() {
        let verbose_level = *VERBOSE_LEVEL.get().unwrap_or(&VerboseLevel::Main);
        let filter_str = match verbose_level {
            VerboseLevel::None => "error",
            // Show our logs, hide ort logs
            VerboseLevel::Main => "info,ort=off",
            // Show debug + ort warnings
            VerboseLevel::Additional => "debug,ort=warn",
            // Show everything
            VerboseLevel::All => "trace",
        };
        let env_filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(filter_str));
        let (filter_layer, handle) = reload::Layer::new(env_filter);
        let _ = RELOAD_HANDLE.set(handle);
        tracing_subscriber::registry()
            .with(filter_layer)
            .with(
                tracing_fmt::layer()
                    .json()
                    // Hide target again - we don't need it now
                    .with_target(false)
                    .with_thread_ids(false)
                    .with_thread_names(false)
                    .with_file(false)
                    .with_line_number(false),
            )
            .init();
    }
}

pub fn set_verbose_level(level: VerboseLevel) {
    let _ = VERBOSE_LEVEL.set(level);
    init_logger();
    if let Some(handle) = RELOAD_HANDLE.get() {
        let filter_str = match level {
            VerboseLevel::None => "error",
            VerboseLevel::Main => "info,ort=off",
            VerboseLevel::Additional => "debug,ort=warn",
            VerboseLevel::All => "trace",
        };
        let _ = handle.modify(|f| {
            *f = EnvFilter::new(filter_str);
        });
    }
}

pub fn get_verbose_level() -> VerboseLevel {
    *VERBOSE_LEVEL.get().unwrap_or(&VerboseLevel::None)
}

pub fn is_verbose_level(level: VerboseLevel) -> bool {
    get_verbose_level() >= level
}

pub fn verbose_log(level: VerboseLevel, event: &str, message: &str) {
    if !is_verbose_level(level) {
        return;
    }

    match level {
        VerboseLevel::None => {}
        VerboseLevel::Main => {
            info!(event = event, message);
        }
        VerboseLevel::Additional => {
            debug!(event = event, message);
        }
        VerboseLevel::All => {
            trace!(event = event, message);
        }
    }
}

pub fn verbose_log_with_fields(
    level: VerboseLevel,
    event: &str,
    message: &str,
    fields: &[(&str, &dyn fmt::Display)],
) {
    if !is_verbose_level(level) {
        return;
    }

    let mut field_map = std::collections::HashMap::new();
    for (key, value) in fields {
        field_map.insert(*key, format!("{}", value));
    }

    match level {
        VerboseLevel::None => {}
        VerboseLevel::Main => {
            info!(event = event, ?field_map, message);
        }
        VerboseLevel::Additional => {
            debug!(event = event, ?field_map, message);
        }
        VerboseLevel::All => {
            trace!(event = event, ?field_map, message);
        }
    }
}

impl VerboseLevel {
    pub fn log(self, event: &str, message: &str) {
        if self == VerboseLevel::None {
            return;
        }
        match self {
            VerboseLevel::None => {}
            VerboseLevel::Main => info!(event = event, message),
            VerboseLevel::Additional => debug!(event = event, message),
            VerboseLevel::All => trace!(event = event, message),
        }
    }

    pub fn log_with_fields(self, event: &str, message: &str, fields: &[(&str, &dyn fmt::Display)]) {
        if self == VerboseLevel::None {
            return;
        }
        let mut field_map = std::collections::HashMap::new();
        for (key, value) in fields {
            field_map.insert(*key, format!("{}", value));
        }
        match self {
            VerboseLevel::None => {}
            VerboseLevel::Main => info!(event = event, ?field_map, message),
            VerboseLevel::Additional => debug!(event = event, ?field_map, message),
            VerboseLevel::All => trace!(event = event, ?field_map, message),
        }
    }

    pub fn is_at_least(self, min_level: VerboseLevel) -> bool {
        self >= min_level
    }
}

#[macro_export]
macro_rules! verbose_log {
	($level:expr, $event:expr, $msg:literal) => {
		$crate::verbose::verbose_log($level, $event, $msg)
	};
	($level:expr, $event:expr, $msg:literal, $($key:literal => $value:expr),+) => {
		$crate::verbose::verbose_log_with_fields(
			$level,
			$event,
			$msg,
			&[$(($key, &$value)),+]
		)
	};
}

#[macro_export]
macro_rules! log_main {
	($event:expr, $msg:literal) => {
		if $crate::verbose::is_verbose_level($crate::verbose::VerboseLevel::Main) {
			tracing::info!(event = $event, $msg);
		}
	};
	($event:expr, $msg:literal, $($key:ident = $value:expr),+) => {
		if $crate::verbose::is_verbose_level($crate::verbose::VerboseLevel::Main) {
			tracing::info!(
				event = $event,
				$($key = $value,)+
				$msg
			);
		}
	};
}

#[macro_export]
macro_rules! log_additional {
	($event:expr, $msg:literal) => {
		if $crate::verbose::is_verbose_level($crate::verbose::VerboseLevel::Additional) {
			tracing::debug!(event = $event, $msg);
		}
	};
	($event:expr, $msg:literal, $($key:ident = $value:expr),+) => {
		if $crate::verbose::is_verbose_level($crate::verbose::VerboseLevel::Additional) {
			tracing::debug!(
				event = $event,
				$($key = $value,)+
				$msg
			);
		}
	};
}

#[macro_export]
macro_rules! log_all {
	($event:expr, $msg:literal) => {
		if $crate::verbose::is_verbose_level($crate::verbose::VerboseLevel::All) {
			tracing::trace!(event = $event, $msg);
		}
	};
	($event:expr, $msg:literal, $($key:ident = $value:expr),+) => {
		if $crate::verbose::is_verbose_level($crate::verbose::VerboseLevel::All) {
			tracing::trace!(
				event = $event,
				$($key = $value,)+
				$msg
			);
		}
	};
}
