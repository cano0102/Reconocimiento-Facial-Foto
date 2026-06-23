"""
Backend en tiempo real para análisis facial con webcam + DeepFace (Flask-SocketIO).

Frontend:
  getUserMedia + canvas
  -> WebSocket -> backend
  -> DeepFace analyze
  -> response por socket
"""

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO
from deepface import DeepFace

# =========================
# CONFIGURACIÓN DE PATHS
# =========================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

if not os.path.isdir(FRONTEND_DIR):
    FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

# =========================
# APP FLASK + SOCKETIO
# =========================

app = Flask(__name__, template_folder=FRONTEND_DIR, static_folder=FRONTEND_DIR)
app.config["SECRET_KEY"] = "deepface-realtime-demo"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="eventlet",  # 🔥 recomendado para Render
    max_http_buffer_size=10_000_000
)

# =========================
# THREADS
# =========================

executor = ThreadPoolExecutor(max_workers=2)
busy_clients = set()

# =========================
# CONFIG DE DEEPFACE
# =========================

ACTIONS = ["emotion", "age", "gender", "race"]
DETECTOR_BACKEND = "opencv"

# =========================
# UTILIDADES
# =========================

def decode_frame(data_url: str) -> np.ndarray:
    """Convierte base64 (dataURL) a imagen OpenCV"""
    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def to_native(obj):
    """Convierte numpy types a Python nativo para JSON"""
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def analyze_frame(frame: np.ndarray):
    """Ejecuta DeepFace"""
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=ACTIONS,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            silent=True
        )

        if isinstance(results, dict):
            results = [results]

        return results

    except Exception as e:
        return {"error": str(e)}

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
def on_connect():
    print("Cliente conectado")


@socketio.on("disconnect")
def on_disconnect():
    from flask import request
    busy_clients.discard(request.sid)
    print("Cliente desconectado")


@socketio.on("frame")
def on_frame(data):
    from flask import request
    sid = request.sid

    if sid in busy_clients:
        return

    busy_clients.add(sid)
    t0 = time.time()

    try:
        frame = decode_frame(data["image"])

    except Exception as e:
        socketio.emit("result", {"error": str(e)}, to=sid)
        busy_clients.discard(sid)
        return

    def task():
        try:
            result = analyze_frame(frame)
            result = to_native(result)

            latency = round((time.time() - t0) * 1000, 2)

            socketio.emit(
                "result",
                {
                    "faces": result,
                    "latency_ms": latency
                },
                to=sid
            )

        except Exception as e:
            socketio.emit("result", {"error": str(e)}, to=sid)

        finally:
            busy_clients.discard(sid)

    executor.submit(task)

# =========================
# PREWARM MODELS
# =========================

def warmup():
    print("Cargando modelos DeepFace...")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    try:
        analyze_frame(dummy)
    except Exception as e:
        print("Warmup error:", e)

# =========================
# MAIN
# =========================

if __name__ == "__main__":
    warmup()

    print("Servidor iniciado")

    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False
    )