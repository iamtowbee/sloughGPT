//! SloughGPT gRPC Service
//! 
//! Async gRPC services for distributed training and inference.
//! 
//! Architecture:
//! - Local: Direct PyO3 calls (fastest)
//! - Distributed: gRPC over network
//! - Streaming: gRPC streaming for federated learning

pub mod trainer_service;
pub mod inference_service;

pub use trainer_service::{TrainerService, GradientAggregator, GradientUpdate};
pub use inference_service::{InferenceService, TokenStreamer};
