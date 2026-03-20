#!/bin/bash
# Custom build script for Vercel to avoid CUDA dependencies

pip install --no-cache-dir \
    numpy==1.24.3 \
    "pillow>=10.0.1" \
    "opencv-python-headless>=4.8.1.78" \
    --only-binary :all:

# Install ultralytics without its heavy dependencies
pip install --no-cache-dir \
    torch==2.0.1 --index-url https://download.pytorch.org/whl/cpu || \
    pip install --no-cache-dir pyyaml requests || true

pip install --no-cache-dir "ultralytics>=8.0.0" --no-deps

echo "✅ Dependencies installed successfully for Vercel"
