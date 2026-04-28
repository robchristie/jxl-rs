use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum Error {
    #[error("input is too short")]
    Truncated,
    #[error("invalid JPEG XL signature")]
    InvalidSignature,
    #[error("invalid JPEG XL container: {0}")]
    InvalidContainer(&'static str),
    #[error("invalid JPEG XL codestream: {0}")]
    InvalidCodestream(&'static str),
    #[error("unsupported JPEG XL feature: {0}")]
    Unsupported(&'static str),
}
