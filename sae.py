"""
Sparse Autoencoder for Analyzing ONNX Driving Model in FrogPilot

This script implements a sparse autoencoder to analyze the ONNX driving model used in FrogPilot.
The goal is to identify specific layers or granular objects within the model graph that correspond
to particular driving actions such as gas and brake events, lateral actions, desires, etc.

Author: FrogPilot Development Team
Date: 2024-04-27
"""

import onnx
import numpy as np
from onnx import numpy_helper
from sklearn.decomposition import SparsePCA
import matplotlib.pyplot as plt

# Constants
DRIVING_MODEL_PATH = "G:/sae/supercombo.onnx"
SPARSE_COMPONENTS = 50  # Number of sparse components for the autoencoder
SPARSE_ALPHA = 1.0      # Sparsity controlling parameter

def load_onnx_model(model_path: str) -> onnx.ModelProto:
    """
    Load the ONNX model from the given path.

    Args:
        model_path (str): URL or local path to the ONNX model.

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

def simulate_layer_outputs(layer_outputs: dict, num_samples: int = 1000) -> dict:
    """
    Simulate realistic output data for each layer for demonstration purposes.
    In a production environment, replace this with actual data collection.

    Args:
        layer_outputs (dict): Dictionary of layer names.
        num_samples (int): Number of samples to simulate.

    Returns:
        dict: Dictionary mapping layer names to simulated numpy arrays.
    """
    np.random.seed(42)  # For reproducibility
    simulated_data = {}
    for layer in layer_outputs:
        # Simulate different shapes based on layer name patterns
        if "fc" in layer or "dense" in layer:
            simulated_data[layer] = np.random.rand(num_samples, 64)
        elif "conv" in layer:
            simulated_data[layer] = np.random.rand(num_samples, 128, 8, 8)
        else:
            simulated_data[layer] = np.random.rand(num_samples, 32)
    print("Simulated layer outputs.")
    return simulated_data

def prepare_data_for_autoencoder(simulated_data: dict) -> np.ndarray:
    """
    Flatten and concatenate layer outputs to form the dataset for the autoencoder.

    Args:
        simulated_data (dict): Simulated layer outputs.

    Returns:
        np.ndarray: Prepared dataset.
    """
    data_list = []
    for layer, data in simulated_data.items():
        data_flat = data.reshape(data.shape[0], -1)  # Flatten spatial dimensions
        data_list.append(data_flat)
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
    sparse_pca = SparsePCA(n_components=n_components, alpha=alpha, random_state=42)
    sparse_pca.fit(data)
    print("Training completed.")
    return sparse_pca

def analyze_components(sparse_pca: SparsePCA, layer_names: list):
    """
    Analyze and visualize the learned sparse components.

    Args:
        sparse_pca (SparsePCA): Trained SparsePCA model.
        layer_names (list): List of layer names corresponding to dataset features.
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

def map_components_to_actions(sparse_pca: SparsePCA, layer_names: list):
    """
    Map the learned components to specific driving actions based on layer associations.

    Args:
        sparse_pca (SparsePCA): Trained SparsePCA model.
        layer_names (list): List of layer names.

    Returns:
        dict: Mapping of components to driving actions.
    """
    component_mapping = {}
    for idx, component in enumerate(sparse_pca.components_):
        # Example heuristic: associate components with layers containing keywords
        associated_layers = [layer for layer in layer_names if any(keyword in layer for keyword in ["fc", "conv", "dense"])]
        component_mapping[f"Component_{idx+1}"] = associated_layers
    print("Mapped components to actions based on layer associations.")
    return component_mapping

def main():
    # Step 1: Load the ONNX driving model
    model = load_onnx_model(DRIVING_MODEL_PATH)

    # Step 2: Extract layer outputs
    layer_outputs = extract_layer_outputs(model)

    # Step 3: Simulate layer outputs (Replace with actual data in production)
    simulated_data = simulate_layer_outputs(layer_outputs)

    # Step 4: Prepare data for the autoencoder
    dataset = prepare_data_for_autoencoder(simulated_data)

    # Step 5: Train the Sparse Autoencoder
    sparse_autoencoder = train_sparse_autoencoder(dataset, SPARSE_COMPONENTS, SPARSE_ALPHA)

    # Step 6: Analyze the learned components
    layer_names = list(layer_outputs.keys())
    analyze_components(sparse_autoencoder, layer_names)

    # Step 7: Map components to driving actions
    component_action_mapping = map_components_to_actions(sparse_autoencoder, layer_names)
    for component, layers in component_action_mapping.items():
        print(f"{component} is associated with layers: {', '.join(layers)}")

    # Step 8: Save the mapping for future reference
    with open("component_action_mapping.pkl", "wb") as f:
        pickle.dump(component_action_mapping, f)
    print("Component-to-action mapping saved to 'component_action_mapping.pkl'.")

if __name__ == "__main__":
    main()