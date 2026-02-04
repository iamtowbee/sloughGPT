"""Machine learning pipeline for SloughGPT."""

from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import json
import numpy as np
import logging
from enum import Enum

try:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA
    import pandas as pd
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import joblib
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ModelType(Enum):
    """Machine learning model types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"
    ANOMALY_DETECTION = "anomaly_detection"
    TIME_SERIES = "time_series"
    RECOMMENDATION = "recommendation"


class TrainingStatus(Enum):
    """Training status for ML models."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    VALIDATING = "validating"
    DEPLOYING = "deploying"


@dataclass
class MLModel:
    """Machine learning model metadata."""
    model_id: str
    name: str
    model_type: ModelType
    version: str
    algorithm: str
    hyperparameters: Dict[str, Any]
    training_data: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    status: TrainingStatus = TrainingStatus.PENDING
    model_path: Optional[str] = None
    is_deployed: bool = False
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingJob:
    """ML training job configuration."""
    job_id: str
    model_type: ModelType
    algorithm: str
    hyperparameters: Dict[str, Any]
    data_source: str
    target_column: Optional[str] = None
    features: List[str] = field(default_factory=list)
    validation_split: float = 0.2
    cross_validation_folds: int = 5
    early_stopping: bool = True
    max_epochs: int = 100
    batch_size: int = 32
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TrainingStatus = TrainingStatus.PENDING
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Prediction result structure."""
    model_id: str
    prediction: Any
    confidence: float
    features_used: List[str]
    prediction_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class MLPipeline:
    """Advanced machine learning pipeline for SloughGPT."""
    
    def __init__(self):
        self.models: Dict[str, MLModel] = {}
        self.training_jobs: Dict[str, TrainingJob] = {}
        self.feature_processors: Dict[str, Any] = {}
        self.model_cache: Dict[str, Any] = {}
        self.prediction_queue = asyncio.Queue()
        self.is_training = False
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize ML pipeline components."""
        try:
            # Initialize feature processors
            self.feature_processors = {
                "standard_scaler": StandardScaler(),
                "pca": PCA(n_components=0.95) if HAS_SKLEARN else None,
                "text_vectorizer": None,  # Will be initialized based on data
                "anomaly_detector": IsolationForest(contamination=0.1) if HAS_SKLEARN else None
            }
            
            self.initialized = True
            logging.info("ML Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logging.error(f"Failed to initialize ML pipeline: {e}")
            return False
    
    async def create_model(self, name: str, model_type: ModelType, algorithm: str,
                        hyperparameters: Dict[str, Any], data_source: str,
                        target_column: Optional[str] = None,
                        features: List[str] = None) -> Optional[str]:
        """Create a new ML model."""
        if not self.initialized:
            logging.error("ML pipeline not initialized")
            return None
        
        model_id = f"ml_model_{datetime.now().timestamp()}"
        
        model = MLModel(
            model_id=model_id,
            name=name,
            model_type=model_type,
            version="1.0.0",
            algorithm=algorithm,
            hyperparameters=hyperparameters,
            training_data=data_source,
            target_column=target_column,
            features=features or []
        )
        
        self.models[model_id] = model
        logging.info(f"Created ML model: {model_id}")
        return model_id
    
    async def train_model(self, model_id: str, training_data: List[Dict[str, Any]],
                       early_stopping: bool = True) -> bool:
        """Train an ML model."""
        if model_id not in self.models:
            logging.error(f"Model {model_id} not found")
            return False
        
        model = self.models[model_id]
        model.status = TrainingStatus.RUNNING
        model.updated_at = datetime.now()
        
        try:
            # Convert training data to DataFrame
            if HAS_SKLEARN:
                df = pd.DataFrame(training_data)
                
                # Prepare features and target
                X = df[model.features] if model.features else df.drop(columns=[model.target_column] if model.target_column else [])
                y = df[model.target_column] if model.target_column else None
                
                # Apply feature preprocessing
                X_processed = self._preprocess_features(X, model.hyperparameters.get("preprocessing", []))
                
                # Train based on model type
                if model.model_type == ModelType.CLASSIFICATION:
                    success = await self._train_classification(model, X_processed, y, early_stopping)
                elif model.model_type == ModelType.REGRESSION:
                    success = await self._train_regression(model, X_processed, y, early_stopping)
                elif model.model_type == ModelType.CLUSTERING:
                    success = await self._train_clustering(model, X_processed)
                elif model.model_type == ModelType.ANOMALY_DETECTION:
                    success = await self._train_anomaly_detection(model, X_processed)
                else:
                    success = await self._train_custom(model, X_processed, y, early_stopping)
                
                if success:
                    model.status = TrainingStatus.COMPLETED
                    model.updated_at = datetime.now()
                    logging.info(f"Model {model_id} trained successfully")
                    return True
            
        except Exception as e:
            model.status = TrainingStatus.FAILED
            model.metrics["error"] = str(e)
            logging.error(f"Training failed for model {model_id}: {e}")
        
        return False
    
    async def _train_classification(self, model: MLModel, X: Any, y: Any,
                              early_stopping: bool = True) -> bool:
        """Train classification model."""
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn is required for classification")
        
        try:
            # Select algorithm
            if model.algorithm == "random_forest":
                clf = RandomForestClassifier(
                    n_estimators=model.hyperparameters.get("n_estimators", 100),
                    max_depth=model.hyperparameters.get("max_depth", None),
                    random_state=42
                )
            elif model.algorithm == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                clf = GradientBoostingClassifier(
                    n_estimators=model.hyperparameters.get("n_estimators", 100),
                    learning_rate=model.hyperparameters.get("learning_rate", 0.1),
                    max_depth=model.hyperparameters.get("max_depth", 3)
                )
            else:
                raise ValueError(f"Unsupported algorithm: {model.algorithm}")
            
            # Split data
            if model.target_column:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
            else:
                X_train, X_val, y_train, y_val = X, None, None, None
            
            # Train model
            clf.fit(X_train, y_train)
            
            # Evaluate model
            if y_val is not None:
                y_pred = clf.predict(X_val)
                model.accuracy = accuracy_score(y_val, y_pred)
                model.precision = precision_score(y_val, y_pred, average='weighted')
                model.recall = recall_score(y_val, y_pred, average='weighted')
                model.f1_score = f1_score(y_val, y_pred, average='weighted')
                
                model.metrics = {
                    "val_accuracy": model.accuracy,
                    "val_precision": model.precision,
                    "val_recall": model.recall,
                    "val_f1": model.f1_score
                }
            
            # Save model
            model.model_path = f"models/{model.model_id}.joblib"
            if HAS_JOBLIB:
                joblib.dump(clf, model.model_path)
            
            # Cache model for inference
            self.model_cache[model.model_id] = clf
            
            return True
            
        except Exception as e:
            logging.error(f"Classification training error: {e}")
            return False
    
    async def _train_regression(self, model: MLModel, X: Any, y: Any,
                            early_stopping: bool = True) -> bool:
        """Train regression model."""
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn is required for regression")
        
        try:
            # Select algorithm
            if model.algorithm == "random_forest":
                from sklearn.ensemble import RandomForestRegressor
                reg = RandomForestRegressor(
                    n_estimators=model.hyperparameters.get("n_estimators", 100),
                    max_depth=model.hyperparameters.get("max_depth", None),
                    random_state=42
                )
            elif model.algorithm == "gradient_boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                reg = GradientBoostingRegressor(
                    n_estimators=model.hyperparameters.get("n_estimators", 100),
                    learning_rate=model.hyperparameters.get("learning_rate", 0.1),
                    max_depth=model.hyperparameters.get("max_depth", 3)
                )
            else:
                raise ValueError(f"Unsupported regression algorithm: {model.algorithm}")
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train model
            reg.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = reg.predict(X_val)
            model.metrics = {
                "val_mse": np.mean((y_val - y_pred) ** 2),
                "val_rmse": np.sqrt(np.mean((y_val - y_pred) ** 2)),
                "val_mae": np.mean(np.abs(y_val - y_pred)),
                "val_r2": reg.score(X_val, y_val)
            }
            
            # Save model
            model.model_path = f"models/{model.model_id}.joblib"
            if HAS_JOBLIB:
                joblib.dump(reg, model.model_path)
            
            # Cache model
            self.model_cache[model.model_id] = reg
            
            return True
            
        except Exception as e:
            logging.error(f"Regression training error: {e}")
            return False
    
    async def _train_clustering(self, model: MLModel, X: Any) -> bool:
        """Train clustering model."""
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn is required for clustering")
        
        try:
            # Select algorithm
            if model.algorithm == "kmeans":
                n_clusters = model.hyperparameters.get("n_clusters", 5)
                clf = KMeans(n_clusters=n_clusters, random_state=42)
            else:
                raise ValueError(f"Unsupported clustering algorithm: {model.algorithm}")
            
            # Train model
            clf.fit(X)
            
            # Evaluate model
            model.metrics = {
                "inertia": clf.inertia_,
                "n_clusters": n_clusters,
                "cluster_centers": clf.cluster_centers_.tolist()
            }
            
            # Save model
            model.model_path = f"models/{model.model_id}.joblib"
            if HAS_JOBLIB:
                joblib.dump(clf, model.model_path)
            
            # Cache model
            self.model_cache[model.model_id] = clf
            
            return True
            
        except Exception as e:
            logging.error(f"Clustering training error: {e}")
            return False
    
    async def _train_anomaly_detection(self, model: MLModel, X: Any) -> bool:
        """Train anomaly detection model."""
        if not HAS_SKLEARN:
            raise ImportError("Scikit-learn is required for anomaly detection")
        
        try:
            # Use pre-initialized anomaly detector
            clf = self.feature_processors.get("anomaly_detector")
            
            if clf is None:
                clf = IsolationForest(contamination=0.1, random_state=42)
            
            # Train model
            clf.fit(X)
            
            # Evaluate model
            predictions = clf.predict(X)
            anomaly_count = sum(1 for p in predictions if p == -1)
            
            model.metrics = {
                "anomaly_count": anomaly_count,
                "anomaly_rate": anomaly_count / len(predictions),
                "contamination": 0.1
            }
            
            # Save model
            model.model_path = f"models/{model.model_id}.joblib"
            if HAS_JOBLIB:
                joblib.dump(clf, model.model_path)
            
            # Cache model
            self.model_cache[model.model_id] = clf
            
            return True
            
        except Exception as e:
            logging.error(f"Anomaly detection training error: {e}")
            return False
    
    async def _train_custom(self, model: MLModel, X: Any, y: Any,
                        early_stopping: bool = True) -> bool:
        """Train custom model using PyTorch."""
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for custom models")
        
        try:
            # Simple neural network for demonstration
            input_size = X.shape[1] if hasattr(X, 'shape') else len(X[0])
            
            if model.model_type == ModelType.CLASSIFICATION:
                output_size = len(np.unique(y))
                criterion = nn.CrossEntropyLoss()
            else:
                output_size = 1
                criterion = nn.MSELoss()
            
            # Create model
            class SimpleNet(nn.Module):
                def __init__(self, input_size, output_size):
                    super().__init__()
                    self.fc1 = nn.Linear(input_size, 128)
                    self.fc2 = nn.Linear(128, 64)
                    self.fc3 = nn.Linear(64, output_size)
                    self.dropout = nn.Dropout(0.2)
                
                def forward(self, x):
                    x = torch.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = torch.relu(self.fc2(x))
                    x = self.dropout(x)
                    x = self.fc3(x)
                    return x
            
            net = SimpleNet(input_size, output_size)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
            
            # Training loop (simplified)
            epochs = model.hyperparameters.get("epochs", 100)
            batch_size = model.hyperparameters.get("batch_size", 32)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.LongTensor(y.values if hasattr(y, 'values') else y)
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = net(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 10 == 0:
                    logging.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            model.metrics = {
                "final_loss": loss.item(),
                "epochs_trained": epochs,
                "model_params": sum(p.numel() for p in net.parameters())
            }
            
            # Save model
            model.model_path = f"models/{model.model_id}.pt"
            torch.save(net.state_dict(), model.model_path)
            
            # Cache model
            self.model_cache[model.model_id] = net
            
            return True
            
        except Exception as e:
            logging.error(f"Custom model training error: {e}")
            return False
    
    def _preprocess_features(self, X: Any, preprocessing_steps: List[str]) -> Any:
        """Apply feature preprocessing steps."""
        if not HAS_SKLEARN:
            return X
        
        processed_X = X.copy() if hasattr(X, 'copy') else X
        
        for step in preprocessing_steps:
            if step == "standard_scale":
                scaler = self.feature_processors.get("standard_scaler")
                if scaler:
                    processed_X = scaler.fit_transform(processed_X)
            elif step == "pca":
                pca = self.feature_processors.get("pca")
                if pca:
                    processed_X = pca.fit_transform(processed_X)
        
        return processed_X
    
    async def predict(self, model_id: str, features: List[Dict[str, Any]],
                   return_probabilities: bool = False) -> Optional[PredictionResult]:
        """Make predictions using trained model."""
        if model_id not in self.models:
            logging.error(f"Model {model_id} not found")
            return None
        
        model = self.models[model_id]
        
        if model.model_id not in self.model_cache:
            # Load model from disk
            if model.model_path:
                try:
                    if model.model_path.endswith('.joblib') and HAS_JOBLIB:
                        self.model_cache[model.model_id] = joblib.load(model.model_path)
                    elif model.model_path.endswith('.pt') and HAS_TORCH:
                        # Need model architecture info for PyTorch
                        pass
                except Exception as e:
                    logging.error(f"Failed to load model {model_id}: {e}")
                    return None
            else:
                logging.error(f"Model {model_id} not trained or model path missing")
                return None
        
        try:
            # Convert features to appropriate format
            if HAS_SKLEARN:
                import pandas as pd
                df = pd.DataFrame(features)
                X = df[model.features] if model.features else df
            else:
                X = np.array([[f.get(col, 0) for col in model.features] for f in features])
            
            # Make prediction
            clf = self.model_cache[model.model_id]
            
            if hasattr(clf, 'predict'):
                prediction = clf.predict(X)
            else:
                # PyTorch prediction
                X_tensor = torch.FloatTensor(X)
                clf.eval()
                with torch.no_grad():
                    prediction = clf(X_tensor).numpy()
            
            # Calculate confidence if possible
            confidence = 1.0
            if hasattr(clf, 'predict_proba') and return_probabilities:
                probabilities = clf.predict_proba(X)
                confidence = np.max(probabilities, axis=1)[0]
            
            # For regression, prediction might be array
            if hasattr(prediction, '__len__') and len(prediction) == 1:
                prediction = prediction[0]
            
            return PredictionResult(
                model_id=model_id,
                prediction=prediction,
                confidence=confidence,
                features_used=model.features,
                prediction_time=datetime.now(),
                metadata={"algorithm": model.algorithm}
            )
            
        except Exception as e:
            logging.error(f"Prediction error for model {model_id}: {e}")
            return None
    
    async def batch_predict(self, model_id: str, features_batch: List[List[Dict[str, Any]]],
                        return_probabilities: bool = False) -> List[PredictionResult]:
        """Make batch predictions."""
        results = []
        
        for features in features_batch:
            result = await self.predict(model_id, features, return_probabilities)
            if result:
                results.append(result)
        
        return results
    
    async def evaluate_model(self, model_id: str, test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate model performance on test data."""
        if model_id not in self.models:
            return {"error": "Model not found"}
        
        model = self.models[model_id]
        
        try:
            # Make predictions
            predictions = []
            features_only = []
            actual_values = []
            
            for item in test_data:
                if model.target_column and model.target_column in item:
                    features_only.append({k: v for k, v in item.items() if k in model.features})
                    actual_values.append(item[model.target_column])
                else:
                    features_only.append(item)
            
            if features_only:
                batch_results = await self.batch_predict(model_id, features_only)
                predictions = [r.prediction for r in batch_results]
            
            # Calculate metrics
            evaluation_results = {
                "model_id": model_id,
                "test_samples": len(test_data),
                "predictions_made": len(predictions),
                "evaluation_timestamp": datetime.now().isoformat()
            }
            
            if actual_values and predictions:
                if model.model_type == ModelType.CLASSIFICATION:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                    import numpy as np
                    
                    y_true = np.array(actual_values)
                    y_pred = np.array(predictions)
                    
                    evaluation_results.update({
                        "accuracy": accuracy_score(y_true, y_pred),
                        "precision": precision_score(y_true, y_pred, average='weighted'),
                        "recall": recall_score(y_true, y_pred, average='weighted'),
                        "f1_score": f1_score(y_true, y_pred, average='weighted'),
                        "confusion_matrix": self._calculate_confusion_matrix(y_true, y_pred)
                    })
                
                elif model.model_type == ModelType.REGRESSION:
                    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                    import numpy as np
                    
                    y_true = np.array(actual_values)
                    y_pred = np.array(predictions)
                    
                    evaluation_results.update({
                        "mse": mean_squared_error(y_true, y_pred),
                        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
                        "mae": mean_absolute_error(y_true, y_pred),
                        "r2_score": r2_score(y_true, y_pred)
                    })
            
            return evaluation_results
            
        except Exception as e:
            return {"error": f"Evaluation failed: {e}"}
    
    def _calculate_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray) -> List[List[int]]:
        """Calculate confusion matrix for classification."""
        try:
            from sklearn.metrics import confusion_matrix
            return confusion_matrix(y_true, y_pred).tolist()
        except ImportError:
            # Fallback implementation
            unique_labels = np.unique(np.concatenate([y_true, y_pred]))
            n_labels = len(unique_labels)
            
            matrix = [[0] * n_labels for _ in range(n_labels)]
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            
            for true_label, pred_label in zip(y_true, y_pred):
                true_idx = label_to_idx[true_label]
                pred_idx = label_to_idx[pred_label]
                matrix[true_idx][pred_idx] += 1
            
            return matrix
    
    async def deploy_model(self, model_id: str, deployment_config: Dict[str, Any]) -> bool:
        """Deploy a trained model."""
        if model_id not in self.models:
            logging.error(f"Model {model_id} not found")
            return False
        
        model = self.models[model_id]
        model.status = TrainingStatus.DEPLOYING
        model.updated_at = datetime.now()
        
        try:
            # In production, this would deploy to serving infrastructure
            # For now, we'll just mark as deployed
            model.is_deployed = True
            model.status = TrainingStatus.COMPLETED
            model.metrics["deployment_config"] = deployment_config
            model.metrics["deployment_time"] = datetime.now().isoformat()
            
            logging.info(f"Model {model_id} deployed successfully")
            return True
            
        except Exception as e:
            model.status = TrainingStatus.FAILED
            model.metrics["deployment_error"] = str(e)
            logging.error(f"Model deployment failed: {e}")
            return False
    
    async def get_model_performance_stats(self, model_id: str) -> Dict[str, Any]:
        """Get performance statistics for a model."""
        if model_id not in self.models:
            return {"error": "Model not found"}
        
        model = self.models[model_id]
        
        return {
            "model_id": model_id,
            "name": model.name,
            "type": model.model_type.value,
            "algorithm": model.algorithm,
            "status": model.status.value,
            "is_deployed": model.is_deployed,
            "accuracy": model.accuracy,
            "precision": model.precision,
            "recall": model.recall,
            "f1_score": model.f1_score,
            "training_metrics": model.metrics,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat()
        }
    
    async def list_models(self, model_type: Optional[ModelType] = None,
                        status: Optional[TrainingStatus] = None) -> List[MLModel]:
        """List models with optional filtering."""
        models = list(self.models.values())
        
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        if status:
            models = [m for m in models if m.status == status]
        
        return models
    
    async def delete_model(self, model_id: str) -> bool:
        """Delete a model and its artifacts."""
        if model_id not in self.models:
            return False
        
        model = self.models[model_id]
        
        try:
            # Remove from cache
            if model_id in self.model_cache:
                del self.model_cache[model_id]
            
            # Remove model file
            if model.model_path:
                import os
                if os.path.exists(model.model_path):
                    os.remove(model.model_path)
            
            # Remove from models
            del self.models[model_id]
            
            logging.info(f"Model {model_id} deleted successfully")
            return True
            
        except Exception as e:
            logging.error(f"Model deletion failed: {e}")
            return False
    
    async def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get overall ML pipeline statistics."""
        models_by_type = {}
        models_by_status = {}
        
        for model in self.models.values():
            model_type = model.model_type.value
            status = model.status.value
            
            models_by_type[model_type] = models_by_type.get(model_type, 0) + 1
            models_by_status[status] = models_by_status.get(status, 0) + 1
        
        return {
            "total_models": len(self.models),
            "models_by_type": models_by_type,
            "models_by_status": models_by_status,
            "deployed_models": sum(1 for m in self.models.values() if m.is_deployed),
            "cached_models": len(self.model_cache),
            "pipeline_initialized": self.initialized,
            "supported_algorithms": [
                "random_forest",
                "gradient_boosting",
                "kmeans",
                "isolation_forest",
                "custom_neural_network"
            ] if HAS_SKLEARN else [],
            "timestamp": datetime.now().isoformat()
        }


# Global ML pipeline manager
ml_pipeline = MLPipeline()