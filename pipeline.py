"""
MLflow-based Pipeline Orchestration
This mimics Kubeflow pipeline structure but uses MLflow for tracking
"""
import os
import mlflow
import mlflow.sklearn
from pathlib import Path
from src.pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pipeline configuration
PIPELINE_NAME = "boston-housing-ml-pipeline"
PIPELINE_DESCRIPTION = "MLOps pipeline for Boston Housing price prediction using MLflow (Kubeflow alternative)"

def boston_housing_pipeline(
    data_path: str = "data/raw/housing.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    learning_rate: float = 0.05,
    max_depth: int = 2,
    n_estimators: int = 100
):
    """
    Complete MLOps pipeline for Boston Housing prediction
    This function orchestrates all components similar to Kubeflow pipeline
    
    Args:
        data_path: Path to the raw dataset
        test_size: Proportion of test set
        random_state: Random seed for reproducibility
        learning_rate: Learning rate for gradient boosting
        max_depth: Maximum depth of trees
        n_estimators: Number of estimators
    """
    
    # Set MLflow experiment (similar to Kubeflow pipeline name)
    mlflow.set_experiment(PIPELINE_NAME)
    
    # Start main pipeline run
    with mlflow.start_run(run_name="pipeline_run") as parent_run:
        
        # Log pipeline parameters
        mlflow.log_params({
            "data_path": data_path,
            "test_size": test_size,
            "random_state": random_state,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "n_estimators": n_estimators
        })
        
        logger.info("=" * 70)
        logger.info("STARTING MLFLOW PIPELINE (Kubeflow Alternative)")
        logger.info("=" * 70)
        
        try:
            # Step 1: Data Extraction
            logger.info("\n" + "=" * 70)
            logger.info("STEP 1: Data Extraction")
            logger.info("=" * 70)
            extracted_data_path = data_extraction_component(data_path)
            logger.info(f"✓ Data extracted to: {extracted_data_path}")
            
            # Step 2: Data Preprocessing
            logger.info("\n" + "=" * 70)
            logger.info("STEP 2: Data Preprocessing")
            logger.info("=" * 70)
            preprocessed_data = data_preprocessing_component(
                extracted_data_path,
                test_size=test_size,
                random_state=random_state
            )
            logger.info("✓ Data preprocessed")
            logger.info(f"  Train data: {preprocessed_data['train_data']}")
            logger.info(f"  Test data: {preprocessed_data['test_data']}")
            
            # Step 3: Model Training
            logger.info("\n" + "=" * 70)
            logger.info("STEP 3: Model Training")
            logger.info("=" * 70)
            model_artifacts = model_training_component(
                preprocessed_data['train_data'],
                preprocessed_data['train_target'],
                learning_rate=learning_rate,
                max_depth=max_depth,
                n_estimators=n_estimators
            )
            logger.info("✓ Model trained")
            logger.info(f"  Model: {model_artifacts['model']}")
            logger.info(f"  Scaler: {model_artifacts['scaler']}")
            
            # Step 4: Model Evaluation
            logger.info("\n" + "=" * 70)
            logger.info("STEP 4: Model Evaluation")
            logger.info("=" * 70)
            metrics_path = model_evaluation_component(
                model_artifacts['model'],
                model_artifacts['scaler'],
                preprocessed_data['test_data'],
                preprocessed_data['test_target']
            )
            logger.info("✓ Model evaluated")
            logger.info(f"  Metrics: {metrics_path}")
            
            # Load and display final metrics
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            
            logger.info("\n" + "=" * 70)
            logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            logger.info("=" * 70)
            logger.info("Final Metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value}")
            
            # Log pipeline completion status
            mlflow.log_param("pipeline_status", "completed")
            mlflow.log_metrics(metrics)
            
            return {
                "status": "success",
                "metrics": metrics,
                "model_path": model_artifacts['model'],
                "metrics_path": metrics_path
            }
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            mlflow.log_param("pipeline_status", "failed")
            mlflow.log_param("error", str(e))
            raise

def compile_pipeline():
    """
    Compile pipeline to YAML-like structure for documentation
    MLflow Alternative: Instead of compiling to Kubeflow YAML, we create
    a pipeline definition JSON that documents the pipeline structure.
    """
    pipeline_def = {
        "name": PIPELINE_NAME,
        "description": PIPELINE_DESCRIPTION,
        "components": [
            {
                "name": "data_extraction",
                "type": "component",
                "inputs": ["data_path"],
                "outputs": ["extracted_data_path"]
            },
            {
                "name": "data_preprocessing",
                "type": "component",
                "inputs": ["input_data_path", "test_size", "random_state"],
                "outputs": ["train_data", "test_data", "train_target", "test_target"]
            },
            {
                "name": "model_training",
                "type": "component",
                "inputs": ["train_data_path", "train_target_path", "learning_rate", "max_depth", "n_estimators"],
                "outputs": ["model", "scaler"]
            },
            {
                "name": "model_evaluation",
                "type": "component",
                "inputs": ["model_path", "scaler_path", "test_data_path", "test_target_path"],
                "outputs": ["metrics_path"]
            }
        ],
        "execution_order": [
            "data_extraction",
            "data_preprocessing",
            "model_training",
            "model_evaluation"
        ]
    }
    
    # Save pipeline definition
    os.makedirs("components", exist_ok=True)
    with open("pipeline.yaml", "w") as f:
        import yaml
        yaml.dump(pipeline_def, f, default_flow_style=False, sort_keys=False)
    
    print("✓ Pipeline definition saved to pipeline.yaml")
    return pipeline_def

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "compile":
        # Compile pipeline definition
        compile_pipeline()
        print("\nPipeline definition compiled to pipeline.yaml")
    else:
        # Run the pipeline
        result = boston_housing_pipeline()
        print("\n" + "=" * 70)
        print("Pipeline Execution Summary:")
        print("=" * 70)
        print(f"Status: {result['status']}")
        print(f"Model Path: {result['model_path']}")
        print(f"Metrics Path: {result['metrics_path']}")
        print("\nTo view MLflow dashboard, run:")
        print("  mlflow ui")
        print("Then open: http://localhost:5000")