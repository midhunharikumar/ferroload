//! Error type for ferroload-core.

use std::fmt;

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    Json(serde_json::Error),
    /// A required modality/column/member was not found.
    NotFound(String),
    /// The dataset/manifest is malformed or unsupported.
    Format(String),
    /// The installed reader is too old for this dataset.
    ReaderTooOld { required: u32, have: u32 },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Json(e) => write!(f, "json error: {e}"),
            Error::NotFound(s) => write!(f, "not found: {s}"),
            Error::Format(s) => write!(f, "format error: {s}"),
            Error::ReaderTooOld { required, have } => write!(
                f,
                "reader too old: dataset requires min_reader_version {required}, have {have}"
            ),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

pub type Result<T> = std::result::Result<T, Error>;
