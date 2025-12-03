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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

def load_config():
    with open("config/settings.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Load Serial ID
    if os.path.exists("config/serial_id.txt"):
        with open("config/serial_id.txt", "r") as f:
            config['serial_id'] = f.read().strip()
    else:
        # Fallback or generate
        config['serial_id'] = "unknown_device"
        
    return config

if __name__ == "__main__":
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
    mqtt.engine = engine # Link back for callbacks

    # 4. Start Inference Loop (in background thread)
    inference_thread = threading.Thread(target=engine.start_loop, daemon=True)
    inference_thread.start()

    # 5. Start API Server (Main Thread)
    # Host 0.0.0.0 allows access from local network
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except KeyboardInterrupt:
        engine.stop()
        logger.info("Shutting down...")