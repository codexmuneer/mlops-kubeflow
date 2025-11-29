from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    logger.info("Evaluating model...")
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(rmse),
        'mae': float(mae),
        'r2_score': float(r2)
    }
    
    logger.info(f"Model Metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return metrics

def save_metrics(metrics: dict, metrics_path: str):
    """Save metrics to JSON file"""
    logger.info(f"Saving metrics to {metrics_path}")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logger.info("Metrics saved successfully")

def load_metrics(metrics_path: str) -> dict:
    """Load metrics from JSON file"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics