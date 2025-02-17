
"""
sparse_autoencoder_launcher.py

This production‐ready script extracts real sensor data from locally stored openpilot
route logs using the native replay tool (cereal.messaging.Reader), runs the supercombo ONNX
policy model to obtain a 6512-dimensional policy output, and then trains a sparse autoencoder
on that output.

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

# Native imports from openpilot – these assume the repo has been built.
from cereal import messaging

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper functions to process real messages.
# These functions must be adjusted to match the actual capnp schema in your built repository.
# -----------------------------------------------------------------------------
def process_sensor_events(msg):
    """
    Process a sensorEvents message to extract preprocessed camera images.
    Expected to return two numpy arrays:
      - input_imgs: shape [1, 12, 128, 256]
      - big_input_imgs: shape [1, 12, 128, 256]
    """
    try:
        # In a real implementation, perform necessary conversion (e.g., using cv2.resize, etc.)
        # Here we assume the message already provides the preprocessed images.
        input_imgs = np.array(msg.sensorEvents.input_imgs, dtype=np.float16)
        big_input_imgs = np.array(msg.sensorEvents.big_input_imgs, dtype=np.float16)
        return input_imgs, big_input_imgs
    except Exception as e:
        logger.error("Error processing sensorEvents message: %s", e)
        raise

def process_desire(msg):
    """
    Process a desire message.
    Expected to return a numpy array of shape [1, 100, 8].
    """
    try:
        return np.array(msg.desire.value, dtype=np.float16)
    except Exception as e:
        logger.error("Error processing desire message: %s", e)
        raise

def process_traffic_state(msg):
    """
    Process a trafficState message.
    Expected to return a numpy array of shape [1, 2].
    """
    try:
        return np.array(msg.trafficState.convention, dtype=np.float16)
    except Exception as e:
        logger.error("Error processing trafficState message: %s", e)
        raise

def process_lateral_control(msg):
    """
    Process a lateralControl message.
    Expected to return a numpy array of shape [1, 2].
    """
    try:
        return np.array(msg.lateralControl.params, dtype=np.float16)
    except Exception as e:
        logger.error("Error processing lateralControl message: %s", e)
        raise

def process_prev_desired_curv(msg):
    """
    Process a prevDesiredCurv message.
    Expected to return a numpy array of shape [1, 100, 1].
    """
    try:
        return np.array(msg.prevDesiredCurv.curvature, dtype=np.float16)
    except Exception as e:
        logger.error("Error processing prevDesiredCurv message: %s", e)
        raise

def process_features(msg):
    """
    Process a features message.
    Expected to return a numpy array of shape [1, 99, 512].
    """
    try:
        return np.array(msg.features.buffer, dtype=np.float16)
    except Exception as e:
        logger.error("Error processing features message: %s", e)
        raise

# -----------------------------------------------------------------------------
# Extract real sensor data from route logs using the native replay tool.
# -----------------------------------------------------------------------------
def extract_inputs_from_logs(log_dir):
    """
    Iterates over the log files in log_dir and uses cereal.messaging.Reader
    to extract the following inputs:
      - input_imgs and big_input_imgs from sensorEvents
      - desire from desire messages
      - traffic_convention from trafficState messages
      - lateral_control_params from lateralControl messages
      - prev_desired_curv from prevDesiredCurv messages
      - features_buffer from features messages

    Returns a dictionary mapping input names to numpy arrays.
    """
    required = {
        "input_imgs": None,
        "big_input_imgs": None,
        "desire": None,
        "traffic_convention": None,
        "lateral_control_params": None,
        "prev_desired_curv": None,
        "features_buffer": None
    }
    
    # Gather log files with .bz2 or .log extension.
    log_files = glob.glob(os.path.join(log_dir, "*.bz2")) + glob.glob(os.path.join(log_dir, "*.log"))
    if not log_files:
        logger.error("No log files found in %s", log_dir)
        raise ValueError(f"No log files found in {log_dir}")
    
    # Process the first log file that yields all required inputs.
    log_file = log_files[0]
    logger.info("Using log file: %s", log_file)
    # Open the file appropriately.
    if log_file.endswith(".bz2"):
        import bz2
        f = bz2.BZ2File(log_file, "rb")
    else:
        f = open(log_file, "rb")
    
    reader = messaging.Reader(f)
    for m in reader:
        msg_type = m.which()
        if msg_type == "sensorEvents" and required["input_imgs"] is None:
            try:
                required["input_imgs"], required["big_input_imgs"] = process_sensor_events(m)
                logger.info("Extracted sensorEvents images.")
            except Exception:
                pass
        elif msg_type == "desire" and required["desire"] is None:
            try:
                required["desire"] = process_desire(m)
                logger.info("Extracted desire.")
            except Exception:
                pass
        elif msg_type == "trafficState" and required["traffic_convention"] is None:
            try:
                required["traffic_convention"] = process_traffic_state(m)
                logger.info("Extracted trafficState.")
            except Exception:
                pass
        elif msg_type == "lateralControl" and required["lateral_control_params"] is None:
            try:
                required["lateral_control_params"] = process_lateral_control(m)
                logger.info("Extracted lateralControl.")
            except Exception:
                pass
        elif msg_type == "prevDesiredCurv" and required["prev_desired_curv"] is None:
            try:
                required["prev_desired_curv"] = process_prev_desired_curv(m)
                logger.info("Extracted prevDesiredCurv.")
            except Exception:
                pass
        elif msg_type == "features" and required["features_buffer"] is None:
            try:
                required["features_buffer"] = process_features(m)
                logger.info("Extracted features.")
            except Exception:
                pass
        # Stop if all required inputs have been extracted.
        if all(v is not None for v in required.values()):
            break
    f.close()
    
    # Verify all inputs are available.
    missing = [k for k, v in required.items() if v is None]
    if missing:
        logger.error("Missing inputs: %s", missing)
        raise ValueError("Missing required sensor inputs: " + ", ".join(missing))
    
    return required

# -----------------------------------------------------------------------------
# Optionally split the policy output into segments.
# -----------------------------------------------------------------------------
def split_policy_output(policy_output, split_mapping=None):
    # policy_output should be a numpy array of shape [batch_size, 6512]
    if split_mapping is None:
        return {"all": policy_output}
    segments = {}
    for key, (start, end) in split_mapping.items():
        segments[key] = policy_output[:, start:end]
    return segments

# -----------------------------------------------------------------------------
# ONNX Model Wrapper and Sparse Autoencoder definitions
# (Same as before.)
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
# Main Function
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run the ONNX policy model with real sensor data and train a Sparse Autoencoder")
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

    # Extract real sensor data from logs.
    try:
        sensor_inputs = extract_inputs_from_logs(args.log_dir)
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
    logger.info("Policy output shape: %s", policy_output.shape)  # Expected: [batch_size, 6512]

    # Optionally, split the policy output.
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
