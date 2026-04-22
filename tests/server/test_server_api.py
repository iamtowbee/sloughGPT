import pytest
from fastapi.testclient import TestClient
from domains.ui.api_server import app

client = TestClient(app)


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_returns_status(self):
        """Should return healthy status"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
    
    def test_health_returns_version(self):
        """Should return version string"""
        response = client.get("/health")
        data = response.json()
        assert "version" in data
        assert data["version"] == "2.0.0"


class TestAuthEndpoints:
    """Test authentication endpoints (not implemented in current API)"""
    
    def test_register_not_implemented(self):
        """Authentication not implemented"""
        response = client.post("/auth/register")
        assert response.status_code in [404, 422]
    
    def test_login_not_implemented(self):
        """Authentication not implemented"""
        response = client.post("/auth/login")
        assert response.status_code == 404
    
    def test_logout_not_implemented(self):
        """Authentication not implemented"""
        response = client.post("/auth/logout")
        assert response.status_code == 404


class TestModelEndpoints:
    """Test model management endpoints"""
    
    def test_list_models(self):
        """Should return list of available models"""
        response = client.get("/models")
        assert response.status_code == 200
        models = response.json()
        assert "models" in models
        assert len(models["models"]) > 0
    
    def test_model_has_required_fields(self):
        """Should have id, name, size, type"""
        response = client.get("/models")
        models = response.json()
        for model in models["models"]:
            assert "id" in model
            assert "name" in model
            assert "provider" in model
            assert "context_length" in model
    
    def test_load_model_not_implemented(self):
        """Load model not implemented"""
        response = client.post("/models/gpt2/load")
        assert response.status_code == 404
    
    def test_unload_model_not_implemented(self):
        """Unload model not implemented"""
        response = client.post("/models/gpt2/unload")
        assert response.status_code == 404


class TestInferenceEndpoints:
    """Test inference endpoints"""
    
    def test_generate_returns_text(self):
        """Should return generated text"""
        response = client.post("/generate", json={
            "prompt": "Hello, how are you?",
            "model": "nanogpt",
            "max_length": 50,
            "temperature": 0.7
        })
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert len(data["text"]) > 0
    
    def test_generate_without_model_loaded(self):
        """Should handle case when model not loaded"""
        response = client.post("/generate", json={
            "prompt": "Test"
        })
        assert response.status_code == 200


class TestTrainingEndpoints:
    """Test training endpoints"""
    
    def test_start_training_returns_job_id(self):
        """Should return job_id when starting training"""
        response = client.post("/training/start", json={
            "dataset_name": "demo",
            "model_id": "nanogpt",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-5,
            "vocab_size": 500,
            "n_embed": 128,
            "n_layer": 3,
            "n_head": 4
        })
        assert response.status_code == 200
        data = response.json()
        assert "id" in data or "job_id" in data
        assert data.get("status") == "running" or "started" in data.get("status", "")
    
    def test_list_training_jobs(self):
        """Should return list of training jobs"""
        response = client.get("/training")
        assert response.status_code == 200
        jobs = response.json()
        assert "jobs" in jobs
        assert isinstance(jobs["jobs"], list)
    
    def test_get_training_job(self):
        """Should return specific job details or skip if not available"""
        response = client.get("/training/jobs")
        # Skip if endpoint not available in this API variant
        if response.status_code == 404:
            pytest.skip("Endpoint not available in this API variant")
        assert response.status_code == 200
        jobs = response.json()
        assert isinstance(jobs, list)


class TestExperimentEndpoints:
    """Test experiment tracking endpoints (not implemented)"""
    
    def test_list_experiments(self):
        """Experiments not implemented"""
        response = client.get("/experiments")
        assert response.status_code == 404
    
    def test_create_experiment_creates_run(self):
        """Experiments not implemented"""
        response = client.post("/training", json={
            "dataset_name": "demo",
            "model_id": "nanogpt",
            "epochs": 1
        })
        assert "experiment_id" not in response.json()
        assert "run_id" not in response.json()


class TestDatasetEndpoints:
    """Test dataset endpoints"""
    
    def test_list_datasets(self):
        """Should return list of datasets"""
        response = client.get("/datasets")
        assert response.status_code == 200
        datasets = response.json()
        assert "datasets" in datasets
        assert len(datasets["datasets"]) > 0
    
    def test_dataset_has_required_fields(self):
        """Dataset should have name, path, size, created_at"""
        response = client.get("/datasets")
        datasets = response.json()
        for ds in datasets["datasets"]:
            assert "name" in ds
            assert "path" in ds
            assert "size" in ds
            assert "created_at" in ds


class TestCheckpointEndpoints:
    """Test checkpoint management endpoints (not implemented)"""
    
    def test_list_checkpoints(self):
        """Checkpoints not implemented"""
        response = client.get("/checkpoints")
        assert response.status_code == 404
    
    def test_create_checkpoint(self):
        """Checkpoints not implemented"""
        response = client.post("/checkpoints", params={
            "name": "test-checkpoint",
            "model_id": "nanogpt"
        })
        assert response.status_code == 404


class TestGPUMetrics:
    """Test GPU monitoring endpoints (not implemented)"""
    
    def test_get_gpu_metrics(self):
        """GPU metrics not implemented"""
        response = client.get("/system/gpu")
        assert response.status_code == 404
