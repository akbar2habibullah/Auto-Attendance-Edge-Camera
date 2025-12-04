from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse # Import this
from pydantic import BaseModel
import cv2
import numpy as np
import time

app = FastAPI(title="Edge Attendance API", version="1.0.0")

# Dependency injection placeholder
engine_instance = None 

def generate_frames():
    """Yields frames in MJPEG format"""
    while True:
        if engine_instance and engine_instance.latest_frame is not None:
            # Encode frame to JPEG
            with engine_instance.frame_lock:
                # Copy to avoid tearing while encoding
                frame_copy = engine_instance.latest_frame.copy()
            
            # Reduce quality to 60% to save bandwidth
            ret, buffer = cv2.imencode('.jpg', frame_copy, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame_bytes = buffer.tobytes()
            
            # MJPEG format structure
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # If no frame yet, create a blank black image
            blank = np.zeros((480, 640, 3), np.uint8)
            cv2.putText(blank, "Waiting for camera...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            ret, buffer = cv2.imencode('.jpg', blank)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
        time.sleep(0.05) # Limit to ~20fps for stream

@app.get("/video_feed")
def video_feed():
    """Stream live video from the camera"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/")
def health_check():
    return {"status": "running", "device": engine_instance.config['serial_id']}

@app.get("/logs")
def get_local_logs(limit: int = 20):
    # Fetch from SQLite
    logs = engine_instance.storage.get_unsynced_logs(limit)
    return {"logs": logs}

@app.post("/faces/add")
async def add_face(name: str, file: UploadFile = File(...)):
    """Add a face manually to the local vector DB"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # Detect face
    bboxes, kpss = engine_instance.detector.detect(img, max_num=1)
    if len(kpss) == 0:
        raise HTTPException(status_code=400, detail="No face detected")
        
    # Get embedding
    embedding = engine_instance.recognizer.get_embedding(img, kpss[0])
    
    # Save to DB
    engine_instance.vectordb.add_face(embedding, name)
    engine_instance.vectordb.save()
    
    return {"status": "success", "name": name}