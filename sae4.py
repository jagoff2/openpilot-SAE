# frogpilot_sparse_autoencoder.py

"""
FrogPilot Sparse Autoencoder for Analyzing ONNX Driving Model with Local Route Logs

This script implements a sparse autoencoder to analyze the ONNX driving model used in FrogPilot.
It processes actual openpilot route logs stored locally to identify specific layers or granular
objects within the model graph that correspond to particular driving actions such as gas and brake
events, lateral actions, desires, etc.

Author: FrogPilot Development Team
Date: 2024-04-27
"""

import onnx
import numpy as np
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt
import pickle
import os

# Import LogReader and related classes from FrogPilot's logreader module
# Ensure that the 'cereal' package is correctly structured and 'log' module is available
from cereal import LogMessage  # Updated import based on the new package structure
from openpilot.tools.lib.logreader import LogReader, Route, ReadMode

# Constants
DRIVING_MODEL_PATH = "models/supercombo.onnx"  # Local path to the ONNX driving model
SPARSE_COMPONENTS = 50  # Number of sparse components for the autoencoder
SPARSE_ALPHA = 1.0      # Sparsity controlling parameter
ROUTE_ID = "a2a0ccea32023010|2023-07-27--13-01-19"  # Example route ID; replace with actual route ID

def load_onnx_model(model_path: str) -> onnx.ModelProto:
    """
    Load the ONNX model from the local path.

    Args:
        model_path (str): Local file path to the ONNX model.

    Returns:
        onnx.ModelProto: Loaded ONNX model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"ONNX model not found at {model_path}")
    print(f"Loading ONNX model from {model_path}...")
    model = onnx.load(model_path)
    print("Model loaded successfully.")
    return model

def extract_layer_names(model: onnx.ModelProto) -> list:
    """
    Extract all layer/output names from the ONNX model.

    Args:
        model (onnx.ModelProto): The loaded ONNX model.

    Returns:
        list: List of layer/output names.
    """
    layer_names = []
    for node in model.graph.node:
        for output in node.output:
            layer_names.append(output)
    print(f"Extracted {len(layer_names)} layer/output names from the model.")
    return layer_names

def load_route_logs(route_id: str) -> list:
    """
    Load the paths of rlog.bz2 and rlog.zst files for the given route.

    Args:
        route_id (str): Identifier for the route.

    Returns:
        list: List of paths to compressed rlog files.
    """
    route = Route(route_id)
    log_paths = route.log_paths()
    if not log_paths:
        raise FileNotFoundError(f"No log files found for route ID: {route_id}")
    print(f"Found {len(log_paths)} log files for route {route_id}.")
    return log_paths

def extract_features_from_logs(log_paths: list) -> dict:
    """
    Extract relevant features from the rlog files.

    Args:
        log_paths (list): List of paths to rlog files.

    Returns:
        dict: Dictionary mapping feature names to their extracted numpy arrays.
    """
    layer_features = {
        'vEgo': [],
        'steeringAngleDeg': [],
        'gasPressed': [],
        'brakePressed': [],
        'steeringPressed': [],
        'desiredCruiseSpeed': [],
        'desiredLongControlState': []
        # Add more features as needed
    }

    print("Extracting features from logs...")
    for log_path in log_paths:
        if log_path is None:
            continue
        log_reader = LogReader(log_path, sort_by_time=True)
        for msg in log_reader:
            # Extract 'carState' messages
            if msg.which() == 'carState':
                car_state = msg.carState
                layer_features['vEgo'].append(car_state.vEgo)  # Vehicle speed
                layer_features['steeringAngleDeg'].append(car_state.steeringAngleDeg)  # Steering angle
                layer_features['steeringPressed'].append(car_state.steeringPressed)  # Steering wheel pressed

            # Extract 'controlsState' messages
            elif msg.which() == 'controlsState':
                controls_state = msg.controlsState
                layer_features['gasPressed'].append(controls_state.gasPressed)  # Gas pedal pressed
                layer_features['brakePressed'].append(controls_state.brakePressed)  # Brake pedal pressed
                layer_features['desiredCruiseSpeed'].append(controls_state.desiredCruiseSpeed)  # Desired cruise speed
                layer_features['desiredLongControlState'].append(controls_state.desiredLongControlState)  # Desired longitudinal control state

            # Extract 'driverMonitoringState' messages for desires (example)
            elif msg.which() == 'driverMonitoringState':
                desires = msg.driverMonitoringState.desire
                layer_features['desiredLongControlState'].append(desires.longitudinal)  # Desired longitudinal action
                # Add more desires as needed

            # Continue extracting other relevant messages and features

    # Convert lists to numpy arrays and handle missing data
    for feature in layer_features:
        if layer_features[feature]:
            layer_features[feature] = np.array(layer_features[feature])
        else:
            layer_features[feature] = np.array([])  # Handle empty features

    print("Feature extraction completed.")
    return layer_features

def prepare_data_for_autoencoder(layer_features: dict) -> (np.ndarray, list):
    """
    Normalize, flatten, and concatenate layer features to form the dataset for the autoencoder.

    Args:
        layer_features (dict): Extracted layer features.

    Returns:
        tuple: Prepared dataset as a numpy array and list of feature names.
    """
    data_list = []
    feature_names = []
    for feature, data in layer_features.items():
        if data.size == 0:
            continue  # Skip features with no data
        # Normalize features to have zero mean and unit variance
        data_normalized = (data - np.mean(data)) / (np.std(data) + 1e-8)
        data_list.append(data_normalized.reshape(-1, 1))  # Reshape for concatenation
        feature_names.append(feature)

    if not data_list:
        raise ValueError("No features extracted from logs.")

    dataset = np.hstack(data_list)
    print(f"Prepared dataset with shape {dataset.shape}.")
    return dataset, feature_names

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
    plt.figure(figsize=(12, 8))
    plt.imshow(components, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Sparse PCA Components")
    plt.xlabel("Feature Index")
    plt.ylabel("Component Index")
    plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    plt.tight_layout()
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
        # Identify features with high absolute weights in the component
        threshold = 0.5  # Threshold for significant weight
        significant_indices = np.where(np.abs(component) > threshold)[0]
        significant_features = [feature_names[i] for i in significant_indices]

        # Heuristically map features to driving actions
        actions = []
        for feature in significant_features:
            if "gas" in feature:
                actions.append("Gas Pressed")
            if "brake" in feature:
                actions.append("Brake Pressed")
            if "steeringAngle" in feature or "steeringPressed" in feature:
                actions.append("Steering Action")
            if "desiredLongControlState" in feature:
                actions.append("Longitudinal Control State")
            # Add more mappings as needed

        # Remove duplicates and store
        actions = list(set(actions))
        component_mapping[f"Component_{idx + 1}"] = actions if actions else ["Unmapped"]

    print("Mapped components to driving actions based on feature associations.")
    return component_mapping

def main():
    """
    Main function to execute the sparse autoencoder analysis workflow.
    """
    try:
        # Step 1: Load the ONNX driving model
        model = load_onnx_model(DRIVING_MODEL_PATH)

        # Step 2: Extract layer/output names from the model
        layer_names = extract_layer_names(model)

        # Step 3: Load route logs
        log_paths = load_route_logs(ROUTE_ID)

        # Step 4: Extract features from logs
        layer_features = extract_features_from_logs(log_paths)

        # Step 5: Prepare data for the autoencoder
        dataset, feature_names = prepare_data_for_autoencoder(layer_features)

        # Step 6: Train the Sparse Autoencoder
        sparse_autoencoder = train_sparse_autoencoder(dataset, SPARSE_COMPONENTS, SPARSE_ALPHA)

        # Step 7: Analyze the learned components
        analyze_components(sparse_autoencoder, feature_names)

        # Step 8: Map components to driving actions
        component_action_mapping = map_components_to_actions(sparse_autoencoder, feature_names)
        for component, actions in component_action_mapping.items():
            print(f"{component} is associated with actions: {', '.join(actions)}")

        # Step 9: Save the mapping for future reference
        with open("component_action_mapping.pkl", "wb") as f:
            pickle.dump(component_action_mapping, f)
        print("Component-to-action mapping saved to 'component_action_mapping.pkl'.")

    except Exception as e:
        print(f"An error occurred: {e}")
        # In production, integrate with FrogPilot's logging mechanism
        # Example:
        # cloudlog.exception("Sparse Autoencoder Analysis Error")
        # sentry.capture_exception(e)
        raise

if __name__ == "__main__":
    main()
  