from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import io
from PIL import Image
import json
import os
from ultralytics import YOLO

app = Flask(__name__)
CORS(app)

# Initialize YOLO model
try:
    model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt for better accuracy
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Global variable for camera
camera = None

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            camera = cv2.VideoCapture(1)  # Try second camera if first fails
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect_image', methods=['POST'])
def detect_image():
    if model is None:
        return jsonify({'error': 'YOLO model not loaded'}), 500
    
    try:
        # Get image from request
        data = request.json
        image_data = data['image'].split(',')[1]  # Remove data:image/jpeg;base64,
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            print(f"Decoded image bytes: {len(image_bytes)}")
        except Exception as e:
            print(f"Base64 decode error: {e}")
            return jsonify({'error': 'Invalid base64 image data'}), 400
            
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            print(f"Image shape: {image_cv.shape}")
        except Exception as e:
            print(f"Image processing error: {e}")
            return jsonify({'error': 'Invalid image file'}), 400
        
        # Run YOLO detection
        print("Running YOLO detection...")
        results = model(image_cv)
        print("YOLO detection completed")
        
        # Draw boxes on image
        annotated_image = draw_boxes(image_cv, results)
        
        # Convert back to base64
        _, buffer = cv2.imencode('.jpg', annotated_image)
        annotated_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract detection data
        detections = extract_detections(results)
        print(f"Found {len(detections)} detections")
        
        return jsonify({
            'success': True,
            'annotated_image': f'data:image/jpeg;base64,{annotated_base64}',
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        print(f"Detection error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def generate_frames():
    """Generate frames for video streaming"""
    camera = get_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        if model is not None:
            # Run YOLO detection
            results = model(frame, verbose=False)
            frame = draw_boxes(frame, results)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/capture_frame', methods=['POST'])
def capture_frame():
    if model is None:
        return jsonify({'error': 'YOLO model not loaded'}), 500
    
    try:
        camera = get_camera()
        success, frame = camera.read()
        
        if not success:
            return jsonify({'error': 'Failed to capture frame'}), 500
        
        # Run YOLO detection
        results = model(frame)
        
        # Draw boxes on image
        annotated_frame = draw_boxes(frame, results)
        
        # Convert to base64
        _, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Extract detection data
        detections = extract_detections(results)
        
        return jsonify({
            'success': True,
            'image': f'data:image/jpeg;base64,{frame_base64}',
            'detections': detections,
            'count': len(detections)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/start_camera', methods=['POST'])
def start_camera():
    try:
        get_camera()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    try:
        release_camera()
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    print("Starting YOLO Object Detection Server...")
    print("Make sure to install required packages:")
    print("pip install flask flask-cors opencv-python ultralytics pillow")
    
    app.run(debug=True, host='0.0.0.0', port=5000)