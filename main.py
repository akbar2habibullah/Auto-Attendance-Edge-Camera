import yaml
import threading
import uvicorn
import os
import logging
from core.engine import InferenceEngine
from services.local_storage import StorageService
from services.mqtt_client import MQTTService
from services.api import app
import services.api as api_module

# Setup Logging
# We set this up globally so all modules use this format
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("main")

def load_config():
    # Helper to load settings
    if not os.path.exists("config/settings.yaml"):
        # Create default config if missing to prevent crash
        os.makedirs("config", exist_ok=True)
        default_conf = {
            'mqtt': {'broker': '127.0.0.1', 'port': 1883, 'username': '', 'password': '', 'topic_prefix': 'devices'},
            'camera': {'index': 0, 'width': 640, 'height': 480},
            'system': {'direction': 'IN', 'debounce_seconds': 10, 'similarity_threshold': 0.45, 'confidence_threshold': 0.5}
        }
        with open("config/settings.yaml", "w") as f:
            yaml.dump(default_conf, f)
            
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load Serial ID
    if os.path.exists("config/serial_id.txt"):
        with open("config/serial_id.txt", "r") as f:
            config['serial_id'] = f.read().strip()
    else:
        config['serial_id'] = "unknown_device"
        
    return config

if __name__ == "__main__":
    try:
        config = load_config()
        logger.info(f"Starting Edge Service for Device: {config['serial_id']}")

        # 1. Initialize Services
        storage = StorageService()
        mqtt = MQTTService(config)
        
        # 2. Initialize AI Engine
        engine = InferenceEngine(config, storage, mqtt)
        
        # Inject engine into API so endpoints can access detection logic
        api_module.engine_instance = engine
        
        # 3. Start MQTT
        mqtt.start()
        mqtt.engine = engine 

        # 4. Start Inference Loop (in background thread)
        inference_thread = threading.Thread(target=engine.start_loop, daemon=True)
        inference_thread.start()

        # 5. Start API Server (Main Thread)
        # FIX: Added log_config=None to prevent crash
        uvicorn.run(app, host="0.0.0.0", port=8000, log_config=None)
        
    except KeyboardInterrupt:
        logger.info("Stopping services...")
        if 'engine' in locals():
            engine.stop()
    except Exception as e:
        logger.critical(f"System Crash: {e}", exc_info=True)