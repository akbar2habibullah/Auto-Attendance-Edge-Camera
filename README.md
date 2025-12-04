# Edge Attendance System (RKNN)

A passive, edge-based face attendance system designed for Rockchip SBCs (RK3566/RK3588). It performs real-time face detection and recognition locally and syncs attendance logs to a cloud server via MQTT.

## 🌟 Features

- **Passive Scanning**: Records attendance automatically as people walk past the camera.
- **Edge Processing**: All AI inference runs locally on the NPU (Neural Processing Unit). No images are sent to the cloud, only metadata.
- **Offline Capable**: Buffers logs locally (SQLite) when the internet is down and syncs when online.
- **Debouncing**: Prevents duplicate logs if a person stands in front of the camera.
- **Dual Interface**: 
  - **MQTT**: For cloud/server integration.
  - **REST API**: For local management and live video streaming.
- **Identity Management**: Unique `serial_id` per device for multi-tenant SaaS deployment.

## 🛠️ Hardware Requirements

- **Platform**: Rockchip RK3566 (Orange Pi 3B, Radxa Zero 3) or RK3588 (Orange Pi 5).
- **OS**: Ubuntu 20.04/22.04 or Debian (Rockchip simplified kernel).
- **Camera**: USB Webcam (MJPEG support recommended) or MIPI CSI Camera.

## 📂 Project Structure

```text
edge-attendance-system/
├── config/              # Configuration files
├── core/                # AI Inference Engine (SCRFD + ArcFace)
├── data/                # Vector DB and SQLite logs
├── services/            # API, MQTT, and Storage logic
├── weights/             # RKNN Model files
├── main.py              # Entry point
└── README.md
```

## 🚀 Installation

### 1. System Dependencies
```bash
sudo apt update
sudo apt install -y python3-pip python3-dev libopencv-dev mosquitto-clients
```

### 2. Python Dependencies
```bash
pip3 install -r requirements.txt
```

> **Note on RKNN**: You must install the `rknn-toolkit-lite2` wheel specific to your board and Python version (usually pre-installed on vendor OS, or available in the vendor's download page).

### 3. Model Weights
Place your converted `.rknn` models in the `weights/` directory:
- `weights/det_10g.rknn` (Detection)
- `weights/w600k_mbf.rknn` (Recognition)

### 4. Device Identity
Create a unique ID for this device. This separates data in the MQTT broker.
```bash
echo "device_001_lobby" > config/serial_id.txt
```

## ⚙️ Configuration

Edit `config/settings.yaml`:

```yaml
mqtt:
  broker: "mqtt.your-cloud.com" # Or localhost for testing
  port: 1883
  username: "device_user"
  password: "secret_password"
  topic_prefix: "devices"

camera:
  index: 0          # /dev/video0
  width: 640
  height: 480

system:
  direction: "IN"   # "IN" or "OUT" (Tags logs with this direction)
  debounce_seconds: 60 # Ignore same face for 60 seconds
  similarity_threshold: 0.45 # Lower = more lenient, Higher = stricter
```

## 🏃 Usage

### Development Mode
Run directly to see logs in the terminal:
```bash
python3 main.py
```

### Production Mode (Systemd)
To run automatically at boot:

1. Copy the service file:
   ```bash
   sudo cp scripts/edge-attendance.service /etc/systemd/system/
   ```
2. Enable and start:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable edge-attendance.service
   sudo systemctl start edge-attendance.service
   ```

---

## 📡 API Reference (Local)

The device runs a local FastAPI server on port **8000**.

**Swagger UI**: `http://<DEVICE_IP>:8000/docs`

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/video_feed` | GET | Stream live annotated video (MJPEG). |
| `/faces/add` | POST | Upload a photo to register a new person. |
| `/faces/list` | GET | List all names stored in the local database. |
| `/faces/debug` | POST | Test a photo against the DB to check similarity scores. |
| `/logs` | GET | Retrieve unsynced attendance logs. |

---

## ☁️ MQTT Protocol Specification

The device communicates with the cloud using JSON payloads.

**Topic Root**: `devices/{serial_id}/`

### 1. Attendance Log (Publish)
**Topic**: `.../attendance`
**Payload**:
```json
{
  "log_id": 105,
  "name": "Akbar",
  "direction": "IN",
  "timestamp": 1701683200.5,
  "device_id": "device_001_lobby"
}
```

### 2. Commands (Subscribe)
**Topic**: `.../commands`

**Command: Sync Faces (Download new DB)**
```json
{
  "type": "sync_faces",
  "data": {
    "url": "https://api.cloud.com/devices/001/faces.zip"
  }
}
```

**Command: Update Config**
```json
{
  "type": "update_config",
  "data": {
    "similarity_threshold": 0.55,
    "debounce_seconds": 30
  }
}
```

## 🔧 Troubleshooting

**1. Face marked as "Unknown" but I am registered**
- Check the debug stream: `http://<IP>:8000/video_feed`
- Use `/faces/debug` to upload your photo.
- If score is between 0.30 - 0.45, lower the `similarity_threshold` in `config/settings.yaml`.

**2. "Query dynamic range failed" Warning**
- This is normal for RKNN static shape models. It is a warning, not an error. Ignore it.

**3. Camera not opening**
- Check permissions: `sudo usermod -aG video $USER`
- Verify index: `ls /dev/video*`. Change `index` in config.

## 📜 License

Private / Proprietary.