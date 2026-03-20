#!/bin/bash

# Vercel Deployment Test Script

echo "🚀 YOLO Object Detection - Vercel Deployment Test"
echo "=================================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python --version

# Check dependencies
echo ""
echo "✓ Checking dependencies..."
pip list | grep -E "(opencv|ultralytics|pillow|numpy)"

# Check file structure
echo ""
echo "✓ Checking project structure..."
echo ""
echo "Required files for Vercel:"
for file in vercel.json api/detect.py public/index.html requirements.txt yolov8n.pt; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "  ✅ $file ($size)"
    else
        echo "  ❌ $file (MISSING)"
    fi
done

echo ""
echo "Optional files:"
for file in VERCEL_DEPLOYMENT.md .vercelignore; do
    if [ -f "$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ⚠️  $file (not found, but optional)"
    fi
done

# Check total size
echo ""
echo "✓ Project size analysis:"
echo "  Total project size: $(du -sh . | awk '{print $1}')"
echo "  Model file: $(ls -lh yolov8n.pt | awk '{print $5}')"
echo "  Deployment limit: 250MB (Vercel)"

# Validate Python syntax
echo ""
echo "✓ Validating Python code..."
python -m py_compile api/detect.py && echo "  ✅ api/detect.py syntax OK" || echo "  ❌ Syntax error in api/detect.py"

echo ""
echo "=================================================="
echo "✅ Deployment readiness check complete!"
echo ""
echo "Next steps:"
echo "1. git add ."
echo "2. git commit -m 'Prepare for Vercel deployment'"
echo "3. vercel --prod"
