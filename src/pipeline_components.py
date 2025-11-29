"""
MLflow-based Pipeline Components
These components mirror Kubeflow component structure but use MLflow for tracking
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import json
import logging
import os
from pathlib import Path
import mlflow
import mlflow.sklearn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================== COMPONENT 1: DATA EXTRACTION ====================
def data_extraction_component(data_path: str, output_dir: str = "data/processed") -> str:
    """
    Component 1: Extract/Load data from DVC-tracked source
    
    INPUT: 
        - data_path (str): Path to the dataset
    
    OUTPUT: 
        - output_path (str): Path where extracted data is saved
    
    MLflow Alternative: Instead of Kubeflow Dataset artifact, we save to file system
    and log the path as an MLflow artifact.
    """
    logger.info(f"[DATA EXTRACTION] Loading data from: {data_path}")
    
    try:
        # Load data (in production, use: dvc get <remote> <path>)
        column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 
                        'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
        data = pd.read_csv(data_path, header=None, delimiter=r"\s+", names=column_names)
        
        logger.info(f"[DATA EXTRACTION] Data loaded successfully. Shape: {data.shape}")
        
        # Save to output path
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "extracted_data.csv")
        data.to_csv(output_path, index=False)
        logger.info(f"[DATA EXTRACTION] Data saved to: {output_path}")
        
        # Log to MLflow (similar to Kubeflow artifact logging)
        with mlflow.start_run(nested=True, run_name="data_extraction"):
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("data_shape", f"{data.shape[0]}x{data.shape[1]}")
            mlflow.log_artifact(output_path, "data")
        
        return output_path
    except Exception as e:
        logger.error(f"[DATA EXTRACTION] Error: {str(e)}")
        raise

# ==================== COMPONENT 2: DATA PREPROCESSING ====================
def data_preprocessing_component(
    input_data_path: str,
    output_dir: str = "data/processed",
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Component 2: Clean, preprocess, and prepare data
    
    INPUT:
        - input_data_path: Path to input dataset
        - test_size: Proportion of test set
        - random_state: Random seed
    
    OUTPUT:
        - dict with paths to: train_data, test_data, train_target, test_target
    
    MLflow Alternative: Instead of multiple Kubeflow Dataset outputs, we return
    a dictionary and log all artifacts to MLflow.
    """
    logger.info("[DATA PREPROCESSING] Starting preprocessing pipeline...")
    
    try:
        # Load data
        data = pd.read_csv(input_data_path)
        logger.info(f"[DATA PREPROCESSING] Initial data shape: {data.shape}")
        
        # Remove outliers
        data = data[~(data['MEDV'] >= 50.0)]
        logger.info(f"[DATA PREPROCESSING] After outlier removal: {data.shape}")
        
        # Feature selection
        column_sels = ['LSTAT', 'INDUS', 'NOX', 'PTRATIO', 'RM', 'TAX', 'DIS', 'AGE']
        X = data[column_sels]
        y = data['MEDV']
        
        # Log transformation for skewed features
        y = np.log1p(y)
        for col in X.columns:
            if np.abs(X[col].skew()) > 0.3:
                logger.info(f"[DATA PREPROCESSING] Applying log transformation to {col}")
                X[col] = np.log1p(X[col])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Save outputs
        os.makedirs(output_dir, exist_ok=True)
        train_data_path = os.path.join(output_dir, "train_data.csv")
        test_data_path = os.path.join(output_dir, "test_data.csv")
        train_target_path = os.path.join(output_dir, "train_target.csv")
        test_target_path = os.path.join(output_dir, "test_target.csv")
        
        X_train.to_csv(train_data_path, index=False)
        X_test.to_csv(test_data_path, index=False)
        pd.DataFrame(y_train).to_csv(train_target_path, index=False)
        pd.DataFrame(y_test).to_csv(test_target_path, index=False)
        
        logger.info(f"[DATA PREPROCESSING] Preprocessing completed.")
        logger.info(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        
        # Log to MLflow
        with mlflow.start_run(nested=True, run_name="data_preprocessing"):
            mlflow.log_params({
                "test_size": test_size,
                "random_state": random_state,
                "train_samples": X_train.shape[0],
                "test_samples": X_test.shape[0],
                "n_features": X_train.shape[1]
            })
            mlflow.log_artifacts(output_dir, "preprocessed_data")
        
        return {
            "train_data": train_data_path,
            "test_data": test_data_path,
            "train_target": train_target_path,
            "test_target": test_target_path
        }
    except Exception as e:
        logger.error(f"[DATA PREPROCESSING] Error: {str(e)}")
        raise

# ==================== COMPONENT 3: MODEL TRAINING ====================
def model_training_component(
    train_data_path: str,
    train_target_path: str,
    models_dir: str = "models",
    learning_rate: float = 0.05,
    max_depth: int = 2,
    n_estimators: int = 100,
    random_state: int = 30
) -> dict:
    """
    Component 3: Train ML model on preprocessed data
    
    INPUT:
        - train_data_path: Path to training features
        - train_target_path: Path to training target
        - learning_rate: Learning rate for gradient boosting
        - max_depth: Maximum depth of trees
        - n_estimators: Number of estimators
    
    OUTPUT:
        - dict with paths to: model, scaler
    
    MLflow Alternative: Instead of Kubeflow Model artifact, we save model and
    log it to MLflow model registry.
    """
    logger.info("[MODEL TRAINING] Starting model training...")
    
    try:
        # Load data
        X_train = pd.read_csv(train_data_path)
        y_train = pd.read_csv(train_target_path).values.ravel()
        
        logger.info(f"[MODEL TRAINING] Data loaded. X shape: {X_train.shape}, y shape: {y_train.shape}")
        
        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        logger.info("[MODEL TRAINING] Features scaled using MinMaxScaler")
        
        # Train model
        logger.info("[MODEL TRAINING] Initializing Gradient Boosting Regressor...")
        model = GradientBoostingRegressor(
            alpha=0.9,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=2,
            n_estimators=n_estimators,
            random_state=random_state
        )
        
        logger.info("[MODEL TRAINING] Training model...")
        model.fit(X_train_scaled, y_train)
        logger.info("[MODEL TRAINING] Model training completed!")
        
        # Save model and scaler
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "trained_model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        
        logger.info(f"[MODEL TRAINING] Model saved to: {model_path}")
        logger.info(f"[MODEL TRAINING] Scaler saved to: {scaler_path}")
        
        # Log to MLflow (similar to Kubeflow model artifact)
        with mlflow.start_run(nested=True, run_name="model_training"):
            mlflow.log_params({
                "learning_rate": learning_rate,
                "max_depth": max_depth,
                "n_estimators": n_estimators,
                "random_state": random_state,
                "model_type": "GradientBoostingRegressor"
            })
            mlflow.sklearn.log_model(model, "model")
            mlflow.log_artifact(scaler_path, "artifacts")
            mlflow.log_artifact(model_path, "artifacts")
        
        return {
            "model": model_path,
            "scaler": scaler_path
        }
    except Exception as e:
        logger.error(f"[MODEL TRAINING] Error: {str(e)}")
        raise

# ==================== COMPONENT 4: MODEL EVALUATION ====================
def model_evaluation_component(
    model_path: str,
    scaler_path: str,
    test_data_path: str,
    test_target_path: str,
    artifacts_dir: str = "artifacts"
) -> str:
    """
    Component 4: Evaluate trained model on test data
    
    INPUT:
        - model_path: Path to trained model
        - scaler_path: Path to fitted scaler
        - test_data_path: Path to test features
        - test_target_path: Path to test target
    
    OUTPUT:
        - metrics_path: Path to evaluation metrics JSON file
    
    MLflow Alternative: Instead of Kubeflow Metrics artifact, we log metrics
    to MLflow and save to JSON file.
    """
    logger.info("[MODEL EVALUATION] Starting model evaluation...")
    
    try:
        # Load model and scaler
        trained_model = joblib.load(model_path)
        fitted_scaler = joblib.load(scaler_path)
        logger.info("[MODEL EVALUATION] Model and scaler loaded")
        
        # Load test data
        X_test = pd.read_csv(test_data_path)
        y_test = pd.read_csv(test_target_path).values.ravel()
        logger.info(f"[MODEL EVALUATION] Test data loaded. Shape: {X_test.shape}")
        
        # Scale test data
        X_test_scaled = fitted_scaler.transform(X_test)
        
        # Make predictions
        logger.info("[MODEL EVALUATION] Making predictions on test set...")
        y_pred = trained_model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            "MSE": float(mse),
            "RMSE": float(rmse),
            "MAE": float(mae),
            "R2_Score": float(r2),
            "Test_Samples": int(X_test.shape[0])
        }
        
        logger.info("[MODEL EVALUATION] Metrics calculated:")
        logger.info(f"  - MSE: {mse:.6f}")
        logger.info(f"  - RMSE: {rmse:.6f}")
        logger.info(f"  - MAE: {mae:.6f}")
        logger.info(f"  - R² Score: {r2:.6f}")
        
        # Save metrics
        os.makedirs(artifacts_dir, exist_ok=True)
        metrics_path = os.path.join(artifacts_dir, "evaluation_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"[MODEL EVALUATION] Metrics saved to: {metrics_path}")
        
        # Log to MLflow (similar to Kubeflow metrics logging)
        with mlflow.start_run(nested=True, run_name="model_evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(metrics_path, "artifacts")
        
        return metrics_path
    except Exception as e:
        logger.error(f"[MODEL EVALUATION] Error: {str(e)}")
        raise