import base64
import os
import time
import numpy as np
import cv2
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from deepface import DeepFace

# =========================
# FLASK APP
# =========================

app = Flask(__name__, template_folder="frontend", static_folder="frontend")
app.config["SECRET_KEY"] = "deepface-realtime-demo"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",  # obligatorio para Render + WebSocket
    max_http_buffer_size=10_000_000
)

# =========================
# CONFIG DE DEEPFACE
# =========================

ACTIONS = ["emotion", "age", "gender", "race"]
DETECTOR_BACKEND = "opencv"

# =========================
# CONTROL SIMPLE DE CARGA
# =========================

busy_clients = set()

# =========================
# UTILIDADES
# =========================

def decode_frame(data_url: str):
    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def to_native(obj):
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def analyze_frame(frame):
    result = DeepFace.analyze(
        img_path=frame,
        actions=ACTIONS,
        detector_backend=DETECTOR_BACKEND,
        enforce_detection=False,
        silent=True
    )

    if isinstance(result, dict):
        result = [result]

    return result

# =========================
# ROUTES
# =========================

@app.route("/")
def index():
    return render_template("index.html")

# =========================
# SOCKET EVENTS
# =========================

@socketio.on("connect")
def connect():
    print("Cliente conectado")


@socketio.on("disconnect")
def disconnect():
    busy_clients.discard(request.sid)


@socketio.on("frame")
def handle_frame(data):
    sid = request.sid

    # backpressure simple
    if sid in busy_clients:
        return

    busy_clients.add(sid)
    start = time.time()

    try:
        frame = decode_frame(data["image"])

        result = analyze_frame(frame)
        result = to_native(result)

        latency = round((time.time() - start) * 1000, 2)

        socketio.emit(
            "result",
            {"faces": result, "latency_ms": latency},
            to=sid
        )

    except Exception as e:
        socketio.emit("result", {"error": str(e)}, to=sid)

    finally:
        busy_clients.discard(sid)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    print("Servidor iniciado...")

    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )