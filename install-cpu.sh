#!/bin/bash
# Install dependencies for local development (CPU only, no CUDA)

echo "🚀 Installing YOLO Object Detection (CPU only)"
echo "========================================"
echo ""
echo "This script installs dependencies without CUDA packages."
echo "This matches the Vercel deployment environment."
echo ""

# Use pip3 directly to bypass uv
echo "📦 Installing core dependencies..."
pip3 install --no-cache-dir --only-binary :all: \
    numpy==1.24.3 \
    Pillow==10.0.1 \
    opencv-python-headless==4.8.1.78

echo ""
echo "📦 Installing PyTorch (CPU only)..."
pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 \
    torchvision==0.15.2

echo ""
echo "📦 Installing ultralytics..."
pip3 install --no-cache-dir ultralytics==8.0.200

echo ""
echo "✅ Installation complete!"
echo ""
echo "Test with:"
echo "  python3 -c \"from ultralytics import YOLO; print('✅ YOLO loaded')\""
echo ""
echo "Run original app:"
echo "  python3 app.py  # Flask app with camera support"
echo ""
echo "For Vercel deployment:"
echo "  vercel --prod"
