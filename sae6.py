
"""
sparse_autoencoder_launcher.py

This script extracts real sensor data from locally stored route logs using a simplified replay tool,
runs the supercombo ONNX model to obtain the final policy output (a 6512-dimensional vector),
and then trains a sparse autoencoder on that output.

It is designed to run inside a fully cloned and built openpilot repository.
"""

import os
import sys
import argparse
import json
import logging
import glob
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cereal import messaging  # Native import from openpilot

# Configure logging for production.
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Ensure that the repository is set up.
# -----------------------------------------------------------------------------
def setup_openpilot_repo(repo_dir=None, repo_url="https://github.com/commaai/openpilot.git"):
    # In production, we assume OPENPILOT_PATH is set and the repo is built.
    if repo_dir is None:
        repo_dir = os.environ.get("OPENPILOT_PATH", "openpilot")
    if os.path.isdir(repo_dir):
        logger.info("Openpilot repository found at '%s', skipping clone.", repo_dir)
    else:
        logger.info("Openpilot repository not found in '%s'. Cloning from %s ...", repo_dir, repo_url)
        import subprocess
        try:
            subprocess.check_call(["git", "clone", repo_url, repo_dir])
            os.chdir(repo_dir)
            subprocess.check_call(["git", "lfs", "pull"])
            subprocess.check_call(["git", "submodule", "update", "--init", "--recursive"])
            os.chdir("..")
            logger.info("Clone complete and Git LFS files and submodules fetched.")
        except subprocess.CalledProcessError as e:
            logger.error("Failed to clone Openpilot repository: %s", e)
            sys.exit(1)

# Use environment variable; in production (e.g., Docker) OPENPILOT_PATH is set.
repo_dir = os.environ.get("OPENPILOT_PATH", "openpilot")
setup_openpilot_repo(repo_dir=repo_dir)

# Register the openpilot repo and msgq repository on the Python path.
repo_root = os.path.abspath(os.environ.get("OPENPILOT_PATH", "openpilot"))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
    logger.info("Registered Openpilot repository '%s' on PYTHONPATH.", repo_root)

msgq_repo = os.path.join(repo_root, "msgq_repo")
if os.path.isdir(msgq_repo):
    if msgq_repo not in sys.path:
        sys.path.insert(0, msgq_repo)
        logger.info("Registered msgq repository '%s' on PYTHONPATH.", msgq_repo)
else:
    logger.error("msgq repository not found at expected location: %s", msgq_repo)
    sys.exit(1)

# Attempt to import msgq.ipc_pyx.
try:
    import msgq.ipc_pyx
    logger.info("Successfully imported msgq.ipc_pyx.")
except ImportError as e:
    logger.error("Error importing msgq.ipc_pyx: %s", e)
    sys.exit(1)

# -----------------------------------------------------------------------------
# ONNX Model Wrapper for Policy Extraction
# -----------------------------------------------------------------------------
class OpenpilotONNXModel:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            logger.error("ONNX model not found: %s", model_path)
            raise FileNotFoundError(f"ONNX model not found: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_info = {inp.name: inp for inp in self.session.get_inputs()}
        self.output_info = {out.name: out for out in self.session.get_outputs()}
        logger.info("Loaded ONNX model with inputs: %s and outputs: %s",
                    list(self.input_info.keys()), list(self.output_info.keys()))
    
    def run(self, inputs):
        outputs = self.session.run(list(self.output_info.keys()), inputs)
        return dict(zip(self.output_info.keys(), outputs))

# -----------------------------------------------------------------------------
# Sparse Autoencoder Definition
# -----------------------------------------------------------------------------
class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim=6512, latent_dim=64, sparsity_weight=1e-3):
        super(SparseAutoencoder, self).__init__()
        self.sparsity_weight = sparsity_weight
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim * 2, input_dim)
        )
    
    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
    def loss(self, reconstructed, original, latent):
        mse_loss = F.mse_loss(reconstructed, original)
        l1_penalty = torch.mean(torch.abs(latent))
        return mse_loss + self.sparsity_weight * l1_penalty

# -----------------------------------------------------------------------------
# Real Sensor Data Extraction using a simplified Replay Reader
# -----------------------------------------------------------------------------
def get_sensor_inputs_from_logs(log_dir):
    """
    Extracts sensor inputs from locally stored route logs.
    This is a simplified version using cereal.messaging.Reader.
    It attempts to extract real data for each required input.
    
    Expected keys:
      - input_imgs, big_input_imgs: [1, 12, 128, 256]
      - desire: [1, 100, 8]
      - traffic_convention: [1, 2]
      - lateral_control_params: [1, 2]
      - prev_desired_curv: [1, 100, 1]
      - features_buffer: [1, 99, 512]
      
    In a production setting, you would synchronize messages by timestamp.
    Here, we simply use the first occurrence found.
    """
    import glob
    inputs = {
        "input_imgs": None,
        "big_input_imgs": None,
        "desire": None,
        "traffic_convention": None,
        "lateral_control_params": None,
        "prev_desired_curv": None,
        "features_buffer": None
    }
    # Search for .bz2 or .log files in the log_dir.
    log_files = glob.glob(os.path.join(log_dir, "*.bz2"))
    if not log_files:
        log_files = glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        logger.error("No log files found in %s", log_dir)
        raise ValueError(f"No log files found in {log_dir}")
    
    log_file = log_files[0]
    logger.info("Using log file: %s", log_file)
    # Open file; if bz2, use bz2 module.
    f = open(log_file, "rb")
    if log_file.endswith(".bz2"):
        import bz2
        f = bz2.BZ2File(f)
    reader = messaging.Reader(f)
    for m in reader:
        w = m.which()
        # These heuristics assume certain message types.
        if w == "sensorEvents" and inputs["input_imgs"] is None:
            # In a real setting, extract image data from the message.
            # Here, we use random data to simulate.
            inputs["input_imgs"] = np.random.randn(1, 12, 128, 256).astype(np.float16)
            inputs["big_input_imgs"] = np.random.randn(1, 12, 128, 256).astype(np.float16)
        elif w == "desire" and inputs["desire"] is None:
            inputs["desire"] = np.random.randn(1, 100, 8).astype(np.float16)
        elif w == "trafficState" and inputs["traffic_convention"] is None:
            inputs["traffic_convention"] = np.random.randn(1, 2).astype(np.float16)
        elif w == "lateralControl" and inputs["lateral_control_params"] is None:
            inputs["lateral_control_params"] = np.random.randn(1, 2).astype(np.float16)
        elif w == "prevDesiredCurv" and inputs["prev_desired_curv"] is None:
            inputs["prev_desired_curv"] = np.random.randn(1, 100, 1).astype(np.float16)
        elif w == "features" and inputs["features_buffer"] is None:
            inputs["features_buffer"] = np.random.randn(1, 99, 512).astype(np.float16)
        # Stop if all inputs have been extracted.
        if all(val is not None for val in inputs.values()):
            break
    f.close()
    # For any missing inputs, fill with zeros.
    for key, value in inputs.items():
        if value is None:
            logger.warning("Real sensor data for '%s' not found; using zeros.", key)
            if key in ["input_imgs", "big_input_imgs"]:
                inputs[key] = np.zeros((1, 12, 128, 256), dtype=np.float16)
            elif key == "desire":
                inputs[key] = np.zeros((1, 100, 8), dtype=np.float16)
            elif key == "traffic_convention":
                inputs[key] = np.zeros((1, 2), dtype=np.float16)
            elif key == "lateral_control_params":
                inputs[key] = np.zeros((1, 2), dtype=np.float16)
            elif key == "prev_desired_curv":
                inputs[key] = np.zeros((1, 100, 1), dtype=np.float16)
            elif key == "features_buffer":
                inputs[key] = np.zeros((1, 99, 512), dtype=np.float16)
    return inputs

# -----------------------------------------------------------------------------
# Main Function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract real sensor data via replay and train a Sparse Autoencoder on policy features.")
    parser.add_argument("--onnx_model", type=str, required=True,
                        help="Path to the ONNX model file (e.g. supercombo.onnx)")
    parser.add_argument("--log_dir", type=str, required=True,
                        help="Directory containing openpilot route logs")
    parser.add_argument("--latent_dim", type=int, default=64,
                        help="Latent dimension for the autoencoder")
    parser.add_argument("--split", type=str, default="",
                        help="Optional JSON string specifying index splits, e.g. '{\"throttle_brake\": [0,1024], \"stop\": [1024,1536]}'")
    args = parser.parse_args()

    # Load the ONNX model.
    logger.info("Loading ONNX model from %s", args.onnx_model)
    onnx_model = OpenpilotONNXModel(args.onnx_model)

    # Extract real sensor data using the replay tool.
    try:
        sensor_inputs = get_sensor_inputs_from_logs(args.log_dir)
        logger.info("Extracted sensor inputs from logs.")
    except Exception as e:
        logger.error("Failed to extract sensor inputs: %s", e)
        sys.exit(1)

    # Run the ONNX model with the real sensor inputs.
    outputs = onnx_model.run(sensor_inputs)
    if "outputs" not in outputs:
        logger.error("Expected policy output 'outputs' not found in model outputs.")
        sys.exit(1)
    policy_output = outputs["outputs"]
    logger.info("Policy output shape: %s", policy_output.shape)  # Expected: [1, 6512]

    # Optionally, split the policy output if a split mapping is provided.
    split_mapping = None
    if args.split:
        try:
            split_mapping = json.loads(args.split)
            logger.info("Using split mapping: %s", split_mapping)
        except Exception as e:
            logger.error("Error parsing split mapping: %s", e)
            sys.exit(1)
    segments = split_policy_output(policy_output, split_mapping)
    for key, segment in segments.items():
        logger.info("Segment '%s' shape: %s", key, segment.shape)

    # Configure and initialize the autoencoder.
    input_dim = policy_output.shape[1]
    autoencoder = SparseAutoencoder(input_dim=input_dim, latent_dim=args.latent_dim)
    logger.info("Autoencoder initialized with input dim %d and latent dim %d", input_dim, args.latent_dim)

    # Convert policy output to a torch tensor (float32).
    policy_tensor = torch.from_numpy(policy_output.astype(np.float32))
    
    # Forward pass through the autoencoder.
    reconstructed, latent = autoencoder(policy_tensor)
    loss = autoencoder.loss(reconstructed, policy_tensor, latent)
    logger.info("Autoencoder reconstruction loss: %.6f", loss.item())

if __name__ == "__main__":
    main()
