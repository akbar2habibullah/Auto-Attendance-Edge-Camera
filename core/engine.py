import time
import cv2
import queue
import threading
import logging
from typing import Optional

from .models.scrfd import SCRFD
from .models.arcface import ArcFace
from .database import VectorDB
from services.local_storage import StorageService
from services.mqtt_client import MQTTService

logger = logging.getLogger("core")

class InferenceEngine:
    def __init__(self, config, storage: StorageService, mqtt: MQTTService):
        self.config = config
        self.storage = storage
        self.mqtt = mqtt
        self.running = False
        
        # Load Models
        self.detector = SCRFD(model_path="weights/det_500m.rknn", conf_thres=config['system']['confidence_threshold'])
        self.recognizer = ArcFace(model_path="weights/w600k_r50.rknn")
        self.vectordb = VectorDB(db_path="./data/vectordb")
        
        # Debounce Cache: {name: last_seen_timestamp}
        self.debounce_cache = {}
        self.debounce_lock = threading.Lock()
7
    def _check_debounce(self, name: str) -> bool:
        """Returns True if the person should be logged, False if recently seen."""
        now = time.time()
        threshold = self.config['system']['debounce_seconds']
        
        with self.debounce_lock:
            if name in self.debounce_cache:
                last_seen = self.debounce_cache[name]
                if now - last_seen < threshold:
                    return False # Too soon
            
            # Update timestamp
            self.debounce_cache[name] = now
            return True

    def process_frame(self, frame):
        # 1. Detect
        bboxes, kpss = self.detector.detect(frame, max_num=5)
        if len(bboxes) == 0:
            return

        # 2. Recognize & Identify
        for i, kps in enumerate(kpss):
            # Get embedding
            embedding = self.recognizer.get_embedding(frame, kps)
            
            # Search in Vector DB
            name, similarity = self.vectordb.search(embedding, threshold=self.config['system']['similarity_threshold'])
            
            if name != "Unknown":
                # 3. Check Debounce (Don't spam logs)
                if self._check_debounce(name):
                    direction = self.config['system']['direction']
                    logger.info(f"Attendance: {name} ({direction}) - Sim: {similarity:.2f}")
                    
                    # 4. Save to Local DB (SQLite)
                    log_id = self.storage.add_log(name, direction, similarity)
                    
                    # 5. Publish to MQTT
                    payload = {
                        "log_id": log_id,
                        "name": name,
                        "direction": direction,
                        "timestamp": time.time(),
                        "device_id": self.config['serial_id']
                    }
                    self.mqtt.publish_attendance(payload)

    def start_loop(self):
        self.running = True
        cap = cv2.VideoCapture(self.config['camera']['index'])
        # Set camera props...
        
        logger.info("Starting Inference Loop")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera failed")
                time.sleep(1)
                continue
            
            try:
                self.process_frame(frame)
            except Exception as e:
                logger.error(f"Inference error: {e}")
                
            # Sleep slightly to free up CPU for API/MQTT if needed
            time.sleep(0.01)
            
        cap.release()

    def stop(self):
        self.running = False