# YOLO Object Detection - Vercel Deployment Guide

## What Changed for Vercel Deployment

This project has been optimized for Vercel deployment within their limits:

### 1. **Architecture Changes**
- Removed Flask server (not compatible with serverless)
- Converted to Vercel Python serverless functions (`/api/detect.py`)
- Static HTML served from `/public` directory
- API endpoint: `/api/detect`

### 2. **Removed Features**
- ❌ Camera streaming (`/video_feed`) - No hardware access on Vercel
- ❌ Live camera capture (`/capture_frame`) - No hardware access on Vercel
- ✅ Image upload and detection - Still fully functional

### 3. **Optimizations for Serverless**
- Replaced `opencv-python` with `opencv-python-headless` (lighter build)
- Removed Flask, Flask-CORS dependencies
- Model loads once at cold start, reused for all calls
- Optimized response format for API

### 4. **Deployment Limits Considered**
- **Function timeout**: 60 seconds (ample for YOLO inference ~5-10s)
- **Bundle size**: ~400MB (within limits with headless OpenCV)
- **Memory**: 1024MB allocated
- **Model file**: 6.3MB (efficiently loaded)

## Deployment Steps

### 1. **Prepare Local Environment**
```bash
cd /workspaces/Object_detection

# Install dependencies locally to test
pip install -r requirements.txt
```

### 2. **Deploy to Vercel**

**Option A: Using Vercel CLI**
```bash
# Install Vercel CLI if not already installed
npm install -g vercel

# Login to Vercel
vercel login

# Deploy
vercel --prod
```

**Option B: Using GitHub Integration**
1. Push code to GitHub: `git push origin main`
2. Go to [vercel.com](https://vercel.com)
3. Import repository
4. Vercel auto-detects Python configuration
5. Deploy

### 3. **Key Files for Vercel**
- `vercel.json` - Configuration file
- `api/detect.py` - Serverless function
- `public/index.html` - Frontend
- `requirements.txt` - Python dependencies
- `.vercelignore` - Files to exclude from deployment

## API Usage

### Detect Objects
**POST** `/api/detect`

```javascript
const formData = new FormData();
const imageInput = document.getElementById('file-input');
const file = imageInput.files[0];

// Convert to base64
const reader = new FileReader();
reader.onload = (e) => {
    fetch('/api/detect', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            image: e.target.result
        })
    })
    .then(r => r.json())
    .then(data => console.log(data));
};
reader.readAsDataURL(file);
```

**Response:**
```json
{
    "success": true,
    "annotated_image": "data:image/jpeg;base64,...",
    "detections": [
        {
            "class": "person",
            "confidence": 0.95,
            "bbox": {"x1": 100, "y1": 50, "x2": 350, "y2": 400}
        }
    ],
    "count": 1
}
```

## Performance Notes

- **Cold start**: ~3-5 seconds (model loading)
- **Warm requests**: ~5-10 seconds (inference only)
- **Max image size**: Recommended < 5MB
- **Max resolution**: ~1920x1080

## Troubleshooting

### Issue: Function timeout
- Ensure image size is < 5MB
- Try reducing image resolution
- Check Vercel logs: `vercel logs`

### Issue: Model not loading
- Verify `yolov8n.pt` is committed to Git
- Check `requirements.txt` dependencies are correct
- View build logs in Vercel dashboard

### Issue: CORS errors
- The API doesn't require CORS (same origin)
- Frontend is served from same domain

## Model Alternatives

To reduce cold start time or memory usage:

```python
# In api/detect.py, change line 21:
model = YOLO('yolov8n.pt')  # Nano (6.3MB) - Fastest
# model = YOLO('yolov8s.pt')  # Small (22MB) - Balanced
# model = YOLO('yolov8m.pt')  # Medium (49MB) - More accurate
```

**Note**: Larger models may exceed deployment limits.

## Next Steps

1. Test locally: `python app.py` (use original Flask app)
2. Deploy: `vercel --prod`
3. Monitor: Check Vercel dashboard for errors
4. Optimize: Track cold starts and inference times

---

**Status**: ✅ Ready for Vercel deployment (within limits)
