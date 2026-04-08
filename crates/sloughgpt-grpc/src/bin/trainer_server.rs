//! Trainer Server Binary
//! 
//! Runs the trainer as a gRPC server.

use sloughgpt_grpc::{TrainerService, GradientAggregator};
use sloughgpt_trainer::TrainConfig;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting SloughGPT Trainer Server");

    let config = TrainConfig::default();
    let _service = TrainerService::new(config.clone());
    let _aggregator = GradientAggregator::new(1); // Single node for now

    tracing::info!("Trainer initialized");
    tracing::info!("Config: vocab_size={}, layers={}, dim={}", 
        config.vocab_size, config.num_layers, config.embedding_dim);

    // TODO: Start gRPC server
    // tonic::Server::builder()
    //     .add_service(TrainerServiceServer::new(service))
    //     .serve("[::1]:50051")
    //     .await?;

    tracing::info!("Server ready. Press Ctrl+C to stop.");

    // Keep running
    tokio::signal::ctrl_c().await?;

    tracing::info!("Shutting down...");
    Ok(())
}
