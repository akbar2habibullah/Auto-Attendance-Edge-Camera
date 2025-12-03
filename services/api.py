from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np

app = FastAPI(title="Edge Attendance API", version="1.0.0")

# Dependency injection placeholder
engine_instance = None 

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