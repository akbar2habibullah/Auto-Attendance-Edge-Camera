import paho.mqtt.client as mqtt
import json
import time
import threading

# Configuration
BROKER = "127.0.0.1"
PORT = 1883
SERIAL_ID = "unknown_device" # Must match what is in config/serial_id.txt

# Topics
TOPIC_ATTENDANCE = f"devices/{SERIAL_ID}/attendance"
TOPIC_COMMANDS = f"devices/{SERIAL_ID}/commands"

def on_connect(client, userdata, flags, rc):
    print(f"✅ [CLOUD] Connected to Broker (RC: {rc})")
    # Subscribe to the attendance topic to "spy" on the device
    client.subscribe(TOPIC_ATTENDANCE)
    print(f"📡 [CLOUD] Listening for data on: {TOPIC_ATTENDANCE}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        print(f"\n📩 [CLOUD] RECEIVED LOG:")
        print(f"   Name: {payload.get('name')}")
        print(f"   Direction: {payload.get('direction')}")
        print(f"   Sim: {payload.get('similarity', 0):.2f}")
        print("-" * 30)
    except Exception as e:
        print(f"❌ [CLOUD] Error parsing message: {e}")

def input_loop(client):
    """Loop to send manual commands to the Edge device"""
    print("\ncommands: 'sync' (simulate DB update), 'conf' (update config), 'exit'")
    while True:
        cmd = input("cloud-admin@shell:~$ ").strip()
        
        if cmd == 'sync':
            # Simulate sending a command to update faces
            payload = {
                "type": "sync_faces",
                "data": {"url": "http://some-fake-url.com/faces.zip"}
            }
            client.publish(TOPIC_COMMANDS, json.dumps(payload))
            print(f"🚀 [CLOUD] Sent Sync Command to {TOPIC_COMMANDS}")
            
        elif cmd == 'conf':
            payload = {
                "type": "update_config",
                "data": {"threshold": 0.6}
            }
            client.publish(TOPIC_COMMANDS, json.dumps(payload))
            print(f"🚀 [CLOUD] Sent Config Update")
            
        elif cmd == 'exit':
            break

if __name__ == "__main__":
    # Load Serial ID from file if exists to ensure matching
    try:
        with open("config/serial_id.txt", "r") as f:
            SERIAL_ID = f.read().strip()
            TOPIC_ATTENDANCE = f"devices/{SERIAL_ID}/attendance"
            TOPIC_COMMANDS = f"devices/{SERIAL_ID}/commands"
    except:
        pass

    client = mqtt.Client(client_id="mock_cloud_server")
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(BROKER, PORT, 60)
        
        # Run MQTT loop in background
        loop_thread = threading.Thread(target=client.loop_forever)
        loop_thread.daemon = True
        loop_thread.start()
        
        # Run input loop in main thread
        input_loop(client)
        
    except ConnectionRefusedError:
        print("❌ Could not connect to Mosquitto. Is it running? (sudo systemctl start mosquitto)")