//! Compatibility adapters and boundary-facing APIs.
//!
//! These modules preserve the current neat-python / JS integration surface while
//! the internal Rust core evolves independently.

pub mod js;
pub mod neat_python;
