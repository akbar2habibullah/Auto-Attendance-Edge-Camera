import time
import cv2
import threading
import logging
import numpy as np # Add this

from .models.scrfd import SCRFD
from .models.arcface import ArcFace
from .database import VectorDB
from .utils import draw_bbox_info, draw_bbox_unknown # Import drawing helpers
from services.local_storage import StorageService
from services.mqtt_client import MQTTService

logger = logging.getLogger("core")

class InferenceEngine:
    def __init__(self, config, storage: StorageService, mqtt: MQTTService):
        self.config = config
        self.storage = storage
        self.mqtt = mqtt
        self.running = False
        
        # Models
        self.detector = SCRFD(model_path="weights/det_500m.rknn", conf_thres=config['system']['confidence_threshold'])
        self.recognizer = ArcFace(model_path="weights/w600k_r50.rknn")
        self.vectordb = VectorDB(db_path="data/vectordb")
        
        # Debounce logic
        self.debounce_cache = {}
        self.debounce_lock = threading.Lock()
        
        # Visualization Storage
        self.latest_frame = None # Store the annotated frame here
        self.frame_lock = threading.Lock()

    def _check_debounce(self, name: str) -> bool:
        now = time.time()
        threshold = self.config['system']['debounce_seconds']
        with self.debounce_lock:
            if name in self.debounce_cache:
                if now - self.debounce_cache[name] < threshold:
                    return False
            self.debounce_cache[name] = now
            return True

    def process_frame(self, frame):
        # 1. Detect
        bboxes, kpss = self.detector.detect(frame, max_num=5)
        
        # If no faces, just update latest frame and return
        if len(bboxes) == 0:
            with self.frame_lock:
                self.latest_frame = frame.copy()
            return

        # 2. Iterate faces
        for i, kps in enumerate(kpss):
            bbox = bboxes[i]
            
            # Get embedding
            try:
                embedding = self.recognizer.get_embedding(frame, kps)

                # Search directly to get the raw score
                name_found, score = self.vectordb.search(embedding, threshold=0.0) # threshold 0 to get ANY match
                
                final_name = "Unknown"
                
                # Check against actual config threshold
                real_threshold = self.config['system']['similarity_threshold']
                
                if score > real_threshold and name_found != "Unknown":
                    final_name = name_found
                    color = (0, 255, 0) # Green
                else:
                    color = (0, 0, 255) # Red
                    # LOGGING: Print why it failed
                    if score > 0.1: # Only log if there is a faint resemblance
                        logger.info(f"Unknown Face detected. Best match: {name_found} with score: {score:.2f} (Needs > {real_threshold})")

                # Drawing
                draw_bbox_info(frame, bbox, score, final_name, color)
                
                # Logic for valid attendance
                if final_name != "Unknown":
                     if self._check_debounce(final_name):
                        direction = self.config['system']['direction']
                        logger.info(f"Attendance: {final_name} ({direction}) - Sim: {score:.2f}")
                        
                        log_id = self.storage.add_log(final_name, direction, score)
                        payload = {
                            "log_id": log_id, "name": final_name, 
                            "direction": direction, "timestamp": time.time(),
                            "device_id": self.config['serial_id']
                        }
                        self.mqtt.publish_attendance(payload)

            except Exception as e:
                logger.error(f"Recog error: {e}")

        # 3. Store annotated frame for streaming
        with self.frame_lock:
            self.latest_frame = frame

    def start_loop(self):
        self.running = True
        logger.info(f"Opening Camera Index: {self.config['camera']['index']}")
        
        # Force MJPG for USB cams to get better framerate
        cap = cv2.VideoCapture(self.config['camera']['index'])
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['camera']['width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['camera']['height'])

        logger.info("Starting Inference Loop")
        while self.running:
            ret, frame = cap.read()
            if not ret:
                logger.error("Camera failed to grab frame")
                time.sleep(1)
                continue
            
            try:
                self.process_frame(frame)
            except Exception as e:
                logger.error(f"Inference Loop Error: {e}")
            
            time.sleep(0.01)
            
        cap.release()

    def stop(self):
        self.running = False