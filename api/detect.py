import base64
import json
import cv2
import numpy as np
from PIL import Image
import io
import os
import sys
from ultralytics import YOLO

# Load model once at cold start
try:
    model = YOLO('yolov8n.pt')
except:
    model = None

def draw_boxes(image, results):
    """Draw bounding boxes on image with labels and confidence scores"""
    annotated_image = image.copy()

    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = f"{model.names[cls]}: {conf:.2f}"

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

def extract_detections(results):
    """Extract detection information as JSON"""
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

    if model is None:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': 'YOLO model not loaded'})
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

        # Run YOLO detection
        results = model(image_cv)

        # Draw boxes on image
        annotated_image = draw_boxes(image_cv, results)

        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')

        # Extract detection data
        detections = extract_detections(results)

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
