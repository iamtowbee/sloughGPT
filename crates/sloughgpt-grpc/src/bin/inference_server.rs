//! Inference Server Binary
//! 
//! Runs the inference engine as a gRPC server.

use sloughgpt_grpc::InferenceService;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    tracing::info!("Starting SloughGPT Inference Server");

    let service = InferenceService::new();

    // Health check
    let health = service.health_check().await;
    tracing::info!("Health: healthy={}, version={}", health.healthy, health.version);

    // TODO: Start gRPC server
    // tonic::Server::builder()
    //     .add_service(InferenceServiceServer::new(service))
    //     .serve("[::1]:50052")
    //     .await?;

    tracing::info!("Server ready. Press Ctrl+C to stop.");

    // Keep running
    tokio::signal::ctrl_c().await?;

    tracing::info!("Shutting down...");
    Ok(())
}
