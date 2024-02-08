#!/bin/bash
cd ..

# Create virtual environment
python -m venv cath

# Activate virtual environment
source cath/bin/activate

# Install requirements
pip install openpyxl
pip install opencv-python

# Install PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Deactivate virtual environment
source cath/bin/deactivate.bat
