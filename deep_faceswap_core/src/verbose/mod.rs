//! Logging module
//!
//! Structured logging system for face swap debugging and monitoring.
//!
//! This module provides hierarchical logging levels and structured event tracking
//! using the `tracing` crate with JSON output format.

pub mod verbose;

pub use self::verbose::*;

use std::sync::Once;

static INIT: Once = Once::new();

pub fn ensure_logger_init() {
    INIT.call_once(|| {
        init_logger();
    });
}
