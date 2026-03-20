import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import os

try:
    import onnxruntime as rt
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# For fallback: try to load ultralytics if available
MODEL_PATH = 'yolov8n.pt'
ONNX_PATH = 'yolov8n.onnx'

model = None
session = None

def init_model():
    """Initialize model - try ONNX first, then fallback to ultralytics"""
    global model, session, ONNX_AVAILABLE

    if ONNX_AVAILABLE and os.path.exists(ONNX_PATH):
        try:
            providers = ['CPUExecutionProvider']
            session = rt.InferenceSession(ONNX_PATH, providers=providers)
            return True
        except Exception as e:
            print(f"ONNX loading failed: {e}")

    # Fallback: try ultralytics
    try:
        from ultralytics import YOLO
        model = YOLO(MODEL_PATH)
        return True
    except Exception as e:
        print(f"YOLO loading failed: {e}")
        return False

def draw_boxes(image, detections):
    """Draw bounding boxes on image"""
    annotated_image = image.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(int, [
            detection['bbox']['x1'],
            detection['bbox']['y1'],
            detection['bbox']['x2'],
            detection['bbox']['y2']
        ])

        conf = detection['confidence']
        label = f"{detection['class']}: {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10),
                     (x1 + label_size[0], y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(annotated_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return annotated_image

def detect_with_ultralytics(image_cv):
    """Use ultralytics for detection"""
    if model is None:
        return []

    results = model(image_cv)
    detections = []

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                detections.append({
                    'class': model.names[cls],
                    'confidence': round(conf, 3),
                    'bbox': {
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2)
                    }
                })

    return detections

def handler(request):
    """Vercel serverless function handler"""

    # Initialize model on first call
    global model, session
    if model is None and session is None:
        if not init_model():
            return {
                'statusCode': 500,
                'body': json.dumps({'error': 'Model failed to load'})
            }

    try:
        # Get image from request
        body = json.loads(request.get('body', '{}'))
        image_data = body.get('image', '').split(',')[1] if ',' in body.get('image', '') else body.get('image', '')

        if not image_data:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'No image data provided'})
            }

        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid base64 image data'})
            }

        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        except Exception as e:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid image file'})
            }

        # Run detection
        detections = detect_with_ultralytics(image_cv)

        # Draw boxes on image
        annotated_image = draw_boxes(image_cv, detections)

        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        return {
            'statusCode': 200,
            'headers': {'Content-Type': 'application/json'},
            'body': json.dumps({
                'success': True,
                'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
                'detections': detections,
                'count': len(detections)
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
