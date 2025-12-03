import json
import logging
import paho.mqtt.client as mqtt
import threading

logger = logging.getLogger("mqtt")

class MQTTService:
    def __init__(self, config, engine_ref=None):
        self.config = config
        self.engine = engine_ref # Reference to engine to update faces
        self.client = mqtt.Client(client_id=f"edge_{config['serial_id']}")
        self.client.username_pw_set(config['mqtt']['username'], config['mqtt']['password'])
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message
        
        self.topic_root = f"{config['mqtt']['topic_prefix']}/{config['serial_id']}"
        self.topic_cmd = f"{self.topic_root}/commands"
        self.topic_data = f"{self.topic_root}/attendance"

    def start(self):
        try:
            self.client.connect(self.config['mqtt']['broker'], self.config['mqtt']['port'], 60)
            self.client.loop_start()
        except Exception as e:
            logger.error(f"MQTT Connection failed: {e}")

    def on_connect(self, client, userdata, flags, rc):
        logger.info(f"Connected to MQTT Broker (RC: {rc})")
        client.subscribe(self.topic_cmd)

    def on_message(self, client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            cmd_type = payload.get('type')
            
            if cmd_type == 'sync_faces':
                logger.info("Received Sync Face command")
                # Logic to download new embeddings or faces from URL in payload
                # self.engine.vectordb.update_from_url(payload['data'])
                
            elif cmd_type == 'update_config':
                logger.info("Received Config update")
                # Update local YAML
                
        except Exception as e:
            logger.error(f"Error processing MQTT message: {e}")

    def publish_attendance(self, data):
        self.client.publish(self.topic_data, json.dumps(data), qos=1)