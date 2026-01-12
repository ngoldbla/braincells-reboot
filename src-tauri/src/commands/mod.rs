//! Tauri commands for IPC with the frontend
//!
//! This module exposes Rust functionality to the TypeScript frontend
//! through Tauri's command system.

pub mod inference;
pub mod models;
pub mod settings;

pub use inference::*;
pub use models::*;
pub use settings::*;
