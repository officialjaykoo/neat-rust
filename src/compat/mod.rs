//! Compatibility adapters and boundary-facing APIs.
//!
//! These modules preserve the current config/model export and JS integration
//! surfaces while the internal Rust core evolves independently.

pub mod js;
pub mod neat_format;
