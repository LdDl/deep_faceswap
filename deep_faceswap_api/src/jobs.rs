//! Job management for long-running operations (video processing)

use serde::Serialize;
use utoipa::ToSchema;

#[derive(Debug, Clone, Serialize, ToSchema)]
#[serde(rename_all = "snake_case")]
pub enum JobStatus {
    Queued,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct JobProgress {
    #[schema(example = "processing_frames")]
    pub stage: String,
    #[schema(example = 150)]
    pub current: usize,
    #[schema(example = 300)]
    pub total: usize,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct JobResult {
    #[schema(example = "/home/user/output.mp4")]
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_path: Option<String>,
    /// Arbitrary result payload (e.g. video analyze response)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schema(value_type = Object)]
    pub data: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, ToSchema)]
pub struct JobState {
    #[schema(example = "a1b2c3d4")]
    pub id: String,
    pub status: JobStatus,
    pub progress: JobProgress,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<JobResult>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl JobState {
    pub fn new_queued(id: String) -> Self {
        Self {
            id,
            status: JobStatus::Queued,
            progress: JobProgress {
                stage: "queued".to_string(),
                current: 0,
                total: 0,
            },
            result: None,
            error: None,
        }
    }
}
