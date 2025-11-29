"""
Compile MLflow components to YAML definitions
This creates component definitions similar to Kubeflow component YAMLs
"""
import yaml
import os
from src.pipeline_components import (
    data_extraction_component,
    data_preprocessing_component,
    model_training_component,
    model_evaluation_component
)
import inspect

def create_component_yaml(func, component_name, description):
    """Create YAML definition for a component"""
    sig = inspect.signature(func)
    
    inputs = []
    for param_name, param in sig.parameters.items():
        param_type = "String" if param.annotation == str else "Float" if param.annotation == float else "Integer" if param.annotation == int else "String"
        inputs.append({
            "name": param_name,
            "type": param_type,
            "default": str(param.default) if param.default != inspect.Parameter.empty else None
        })
    
    # Get return annotation
    return_type = sig.return_annotation if sig.return_annotation != inspect.Signature.empty else "String"
    
    component_def = {
        "name": component_name,
        "description": description,
        "implementation": {
            "container": {
                "image": "python:3.9",
                "command": ["python", "-c"],
                "args": [f"# {component_name} implementation"]
            }
        },
        "inputs": inputs,
        "outputs": [
            {
                "name": "output",
                "type": str(return_type).replace("<class '", "").replace("'>", "")
            }
        ]
    }
    
    return component_def

# Create components directory
os.makedirs("components", exist_ok=True)

print("Compiling MLflow components to YAML definitions...")

# Component 1: Data Extraction
comp1 = create_component_yaml(
    data_extraction_component,
    "data_extraction",
    "Extract/Load data from DVC-tracked source"
)
with open("components/data_extraction.yaml", "w") as f:
    yaml.dump(comp1, f, default_flow_style=False, sort_keys=False)
print("✓ Compiled: components/data_extraction.yaml")

# Component 2: Data Preprocessing
comp2 = create_component_yaml(
    data_preprocessing_component,
    "data_preprocessing",
    "Clean, preprocess, and prepare data"
)
with open("components/data_preprocessing.yaml", "w") as f:
    yaml.dump(comp2, f, default_flow_style=False, sort_keys=False)
print("✓ Compiled: components/data_preprocessing.yaml")

# Component 3: Model Training
comp3 = create_component_yaml(
    model_training_component,
    "model_training",
    "Train ML model on preprocessed data"
)
with open("components/model_training.yaml", "w") as f:
    yaml.dump(comp3, f, default_flow_style=False, sort_keys=False)
print("✓ Compiled: components/model_training.yaml")

# Component 4: Model Evaluation
comp4 = create_component_yaml(
    model_evaluation_component,
    "model_evaluation",
    "Evaluate trained model on test data"
)
with open("components/model_evaluation.yaml", "w") as f:
    yaml.dump(comp4, f, default_flow_style=False, sort_keys=False)
print("✓ Compiled: components/model_evaluation.yaml")

print("\n✓ All components compiled successfully!")
print("Component YAML files are in the components/ directory")
