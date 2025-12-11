from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import cv2
import numpy as np
import time
import yaml
import io

app = FastAPI(title="Edge Attendance API", version="1.0.0")

# Mount the static folder for the UI
app.mount("/static", StaticFiles(directory="static"), name="static")

# Dependency injection placeholder
engine_instance = None 
class ZoneConfig(BaseModel):
    points: List[List[int]] # e.g., [[0,0], [100,0], [100,100]]

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

@app.get("/faces/list")
def list_stored_faces():
    """Debug: Show all names currently in the Vector DB"""
    if not engine_instance:
        return {"error": "Engine not initialized"}
    
    # Access metadata directly from the DB
    names = engine_instance.vectordb.metadata
    return {
        "count": len(names),
        "identities": names
    }

@app.post("/faces/debug")
async def debug_face_search(file: UploadFile = File(...)):
    """Debug: Upload a photo and get the raw similarity score, ignoring thresholds"""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    # 1. Detect
    bboxes, kpss = engine_instance.detector.detect(img, max_num=1)
    if len(kpss) == 0:
        return {"status": "fail", "reason": "No face detected in image"}

    # 2. Get Embedding
    embedding = engine_instance.recognizer.get_embedding(img, kpss[0])
    
    # 3. Manual Search (Bypassing the threshold check in vector_db.search)
    # We want to see the top match, even if it's 0.1 (terrible match)
    db = engine_instance.vectordb
    
    if db.index.ntotal == 0:
        return {"status": "fail", "reason": "Database is empty"}

    if embedding.ndim == 1:
        embedding = np.expand_dims(embedding, axis=0)
    
    # Search for top 1
    import faiss # Import locally just for this debug function if needed
    # Ensure normalized (our add_face does this, but good to ensure)
    faiss.normalize_L2(embedding)
    
    D, I = db.index.search(embedding.astype(np.float32), 1)
    
    score = float(D[0][0])
    index_id = int(I[0][0])
    
    matched_name = "Unknown"
    if 0 <= index_id < len(db.metadata):
        matched_name = db.metadata[index_id]

    return {
        "status": "success",
        "best_match_name": matched_name,
        "similarity_score": score,
        "threshold_setting": engine_instance.config['system']['similarity_threshold'],
        "is_match": score > engine_instance.config['system']['similarity_threshold']
    }

@app.get("/snapshot")
def get_snapshot():
    """Returns a single static image for the UI canvas"""
    if engine_instance and engine_instance.latest_frame is not None:
        ret, buffer = cv2.imencode('.jpg', engine_instance.latest_frame)
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    return {"error": "Camera not ready"}

@app.get("/zone")
def get_zone():
    """Get current polygon points"""
    # Convert numpy array back to list
    if engine_instance:
        return {"points": engine_instance.zone_points.tolist()}
    return {"points": []}

@app.post("/zone")
def set_zone(config: ZoneConfig):
    """Update polygon points and save to YAML"""
    if not engine_instance:
        raise HTTPException(status_code=500, detail="Engine not ready")
        
    # 1. Update running engine immediately
    engine_instance.update_zone(config.points)
    
    # 2. Persist to settings.yaml
    try:
        with open("config/settings.yaml", "r") as f:
            yaml_data = yaml.safe_load(f)
        
        yaml_data['system']['zone'] = config.points
        
        with open("config/settings.yaml", "w") as f:
            yaml.dump(yaml_data, f)
            
    except Exception as e:
        print(f"Failed to save config: {e}")
        
    return {"status": "updated", "points": config.points}