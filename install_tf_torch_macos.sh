#!/bin/bash

# This script sets up the environment variables, installs TensorFlow and PyTorch on macOS, and validates the installations.

# Install Xcode Command Line Tools
echo "Installing Xcode Command Line Tools..."
xcode-select --install

# Install TensorFlow on macOS
echo "Installing TensorFlow..."
SYSTEM_VERSION_COMPAT=0 python3 -m pip install tensorflow-macos
SYSTEM_VERSION_COMPAT=0 python3 -m pip install tensorflow-metal
SYSTEM_VERSION_COMPAT=0 python3 -m pip install tensorflow_datasets

# Validate TensorFlow installation
echo "Validating TensorFlow installation..."
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Num GPUs Available: ', len(tf.config.experimental.list_physical_devices('GPU')))"

# Install PyTorch on macOS
echo "Installing PyTorch..."
SYSTEM_VERSION_COMPAT=0 python3 -m pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip install pytorch-lightning

# Validate PyTorch installation
echo "Validating PyTorch installation..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('Is CUDA available:', torch.cuda.is_available())"
python -c "import pytorch_lightning as pl; print('PyTorch Lightning version:', pl.__version__)"
