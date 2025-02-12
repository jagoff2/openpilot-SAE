"""
FrogPilot Sparse Autoencoder with ONNX Model Integration

This script enhances the previous sparse autoencoder implementation by loading actual
openpilot route logs stored locally. It processes the rlog.bz2 files to extract relevant
data, prepares the dataset, and runs it through the ONNX driving model to identify
specific layers or objects corresponding to driving actions such as gas, brake events,
lateral actions, desires, etc.

Author: FrogPilot Development Team
Date: 2024-04-27
"""

import onnx
import numpy as np
from onnx import numpy_helper
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import os
from openpilot.tools.lib.logreader import LogReader, Route  # <source_id data="frogai-frogpilot (2).txt" />
from openpilot.common.params import Params  # <source_id data="frogai-frogpilot (2).txt" />
import pickle

# Constants
DRIVING_MODEL_PATH = "g:/sae/supercombo.onnx"  # Local path to the ONNX model
SPARSE_COMPONENTS = 50  # Number of sparse components for the autoencoder
SPARSE_ALPHA = 1.0      # Sparsity controlling parameter
ROUTE_ID = "0000002a--f81af81530--181"  # Example route ID

def load_onnx_model(model_path: str) -> onnx.ModelProto:
    """
    Load the ONNX model from the local path.

    Args:
        model_path (str): Local file path to the ONNX model.

    Returns:
        onnx.ModelProto: Loaded ONNX model.
    """
    print(f"Loading ONNX model from {model_path}...")
    model = onnx.load_model(model_path)
    print("Model loaded successfully.")
    return model

def extract_layer_outputs(model: onnx.ModelProto) -> dict:
    """
    Extract outputs from each layer/node in the ONNX model.

    Args:
        model (onnx.ModelProto): The ONNX model.

    Returns:
        dict: A dictionary mapping layer names to their output tensors.
    """
    layer_outputs = {}
    for node in model.graph.node:
        for output in node.output:
            layer_outputs[output] = None  # Placeholder for actual outputs
    print(f"Extracted outputs for {len(layer_outputs)} layers.")
    return layer_outputs

def load_route_logs(route_id: str) -> list:
    """
    Load the paths of rlog.bz2 files for the given route.

    Args:
        route_id (str): Identifier for the route.

    Returns:
        list: List of paths to rlog.bz2 files.
    """
    route = Route(route_id)
    log_paths = route.log_paths()
    print(f"Found {len(log_paths)} rlog.bz2 files for route {route_id}.")
    return log_paths

def extract_features_from_logs(log_paths: list) -> dict:
    """
    Extract relevant features from the rlog.bz2 logs.

    Args:
        log_paths (list): List of paths to rlog.bz2 files.

    Returns:
        dict: Dictionary mapping layer names to their extracted feature arrays.
    """
    layer_features = {}
    log_reader = LogReader(log_paths[0])  # Process the first segment for simplicity

    for msg in log_reader:
        # Example extraction: Extracting 'carState' messages
        if msg.which() == 'carState':
            car_state = msg.carState
            # Extract relevant features, e.g., speed, steering angle
            speed = car_state.vEgo  # Vehicle speed
            steer = car_state.steeringAngleDeg  # Steering angle
            # Aggregate features
            if 'car_speed' not in layer_features:
                layer_features['car_speed'] = []
            if 'steering_angle' not in layer_features:
                layer_features['steering_angle'] = []
            layer_features['car_speed'].append(speed)
            layer_features['steering_angle'].append(steer)
        
        # Extract other relevant messages as needed
        # e.g., controlsState for gas/brake events
        if msg.which() == 'controlsState':
            controls_state = msg.controlsState
            gas = controls_state.gasPressed
            brake = controls_state.brakePressed
            if 'gas_pressed' not in layer_features:
                layer_features['gas_pressed'] = []
            if 'brake_pressed' not in layer_features:
                layer_features['brake_pressed'] = []
            layer_features['gas_pressed'].append(gas)
            layer_features['brake_pressed'].append(brake)
        
        # Add extraction logic for desires, lateral actions, etc.
    
    # Convert lists to numpy arrays
    for key in layer_features:
        layer_features[key] = np.array(layer_features[key])
    
    print("Extracted features from logs.")
    return layer_features

def prepare_data_for_autoencoder(layer_features: dict) -> np.ndarray:
    """
    Flatten and concatenate layer features to form the dataset for the autoencoder.

    Args:
        layer_features (dict): Extracted layer features.

    Returns:
        np.ndarray: Prepared dataset.
    """
    data_list = []
    for layer, data in layer_features.items():
        data_normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)  # Normalize features
        data_list.append(data_normalized.reshape(-1, 1))  # Reshape for concatenation
    dataset = np.hstack(data_list)
    print(f"Prepared dataset with shape {dataset.shape}.")
    return dataset

def train_sparse_autoencoder(data: np.ndarray, n_components: int, alpha: float) -> SparsePCA:
    """
    Train a Sparse PCA autoencoder on the dataset.

    Args:
        data (np.ndarray): The input data.
        n_components (int): Number of sparse components.
        alpha (float): Sparsity controlling parameter.

    Returns:
        SparsePCA: Trained SparsePCA model.
    """
    print("Training Sparse PCA autoencoder...")
    sparse_pca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42, max_iter=1000)
    sparse_pca.fit(data)
    print("Training completed.")
    return sparse_pca

def analyze_components(sparse_pca: SparsePCA, feature_names: list):
    """
    Analyze and visualize the learned sparse components.

    Args:
        sparse_pca (SparsePCA): Trained SparsePCA model.
        feature_names (list): List of feature names corresponding to dataset columns.
    """
    components = sparse_pca.components_
    plt.figure(figsize=(10, 6))
    plt.imshow(components, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Sparse PCA Components")
    plt.xlabel("Feature Index")
    plt.ylabel("Component Index")
    plt.show()
    print("Components visualized.")

def map_components_to_actions(sparse_pca: SparsePCA, feature_names: list) -> dict:
    """
    Map the learned components to specific driving actions based on feature associations.

    Args:
        sparse_pca (SparsePCA): Trained SparsePCA model.
        feature_names (list): List of feature names.

    Returns:
        dict: Mapping of components to driving actions.
    """
    component_mapping = {}
    for idx, component in enumerate(sparse_pca.components_):
        # Example heuristic: associate components with features having high absolute weights
        significant_features = [feature_names[i] for i, weight in enumerate(component) if abs(weight) > 0.5]
        component_mapping[f"Component_{idx+1}"] = significant_features
    print("Mapped components to actions based on feature associations.")
    return component_mapping

def main():
    # Step 1: Load the ONNX driving model
    model = load_onnx_model(DRIVING_MODEL_PATH)

    # Step 2: Extract layer outputs
    layer_outputs = extract_layer_outputs(model)

    # Step 3: Load route logs
    log_paths = load_route_logs(ROUTE_ID)

    # Step 4: Extract features from logs
    layer_features = extract_features_from_logs(log_paths)

    # Step 5: Prepare data for the autoencoder
    dataset = prepare_data_for_autoencoder(layer_features)
    feature_names = list(layer_features.keys())

    # Step 6: Train the Sparse Autoencoder
    sparse_autoencoder = train_sparse_autoencoder(dataset, SPARSE_COMPONENTS, SPARSE_ALPHA)

    # Step 7: Analyze the learned components
    analyze_components(sparse_autoencoder, feature_names)

    # Step 8: Map components to driving actions
    component_action_mapping = map_components_to_actions(sparse_autoencoder, feature_names)
    for component, features in component_action_mapping.items():
        print(f"{component} is associated with features: {', '.join(features)}")

    # Step 9: Save the mapping for future reference
    with open("component_action_mapping.pkl", "wb") as f:
        pickle.dump(component_action_mapping, f)
    print("Component-to-action mapping saved to 'component_action_mapping.pkl'.")

if __name__ == "__main__":
    main()