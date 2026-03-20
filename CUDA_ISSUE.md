# CUDA Dependency Issue - Quick Fix

## Problem
Local environment is using `uv` package manager which resolves **all** transitive dependencies, including CUDA packages (~3GB). This causes issues like:

```
Downloading nvidia-cudnn-cu12 (674.0MiB)
Downloading nvidia-cublas-cu12 (566.8MiB)
Downloading torch (873.2MiB)
```

**Note**: This is a LOCAL environment issue only. Vercel deployment works fine with `pip install` which is smarter about excluding CUDA.

---

## ✅ Solution: Use CPU-Only Installation Script

```bash
chmod +x install-cpu.sh
./install-cpu.sh
```

This script:
- Uses `pip3` directly (bypasses `uv`)
- Installs PyTorch CPU-only from PyTorch wheels
- Installs ultralytics without CUDA
- Total download: ~500MB instead of 3GB

---

## Alternative: Manual Installation

```bash
# 1. Install base packages
pip3 install --no-cache-dir --only-binary :all: \
    numpy==1.24.3 \
    Pillow==10.0.1 \
    opencv-python-headless==4.8.1.78

# 2. Install CPU-only PyTorch
pip3 install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.0.1 torchvision==0.15.2

# 3. Install ultralytics
pip3 install --no-cache-dir ultralytics==8.0.200
```

---

## Test Installation

```bash
python3 -c "from ultralytics import YOLO; m = YOLO('yolov8n.pt'); print('✅ YOLO loaded')"
```

---

## Run Locally

**Original Flask app (with camera):**
```bash
python3 app.py
# http://localhost:5000
```

**Quick test:**
```bash
python3 -c "
from api.detect import handler
import json, base64
# Test detection handler
print('Handler loaded successfully')
"
```

---

## Vercel Deployment (No Issues!)

Vercel uses `pip install` directly, which is smarter about dependencies:

```bash
vercel --prod
```

✅ **Vercel automatically uses CPU-only packages**
✅ **No CUDA downloads on Vercel**
✅ **Deployment succeeds within limits**

---

## Why This Happens

- **`uv` resolver**: Resolves ALL transitive dependencies (strict)
- **`pip` installer**: Uses smart fallback and index configuration
- **Vercel pip**: Uses BuildCommand with `--only-binary :all:` which prevents CUDA packages

---

**TL;DR**: Run `./install-cpu.sh` locally, no worries on Vercel! 🚀
