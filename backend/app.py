"""
Backend en tiempo real para análisis facial con webcam + DeepFace.

Arquitectura:
  Frontend (getUserMedia + canvas) --frame base64 (WebSocket)--> Backend (Flask-SocketIO)
  Backend ejecuta DeepFace.analyze() en un thread pool (no bloquea el loop de eventos)
  y devuelve el resultado por el mismo socket.

Por qué WebSocket y no HTTP polling:
  - Evita el overhead de abrir una conexión HTTP nueva por cada frame.
  - Permite "frame dropping" controlado: si el backend está ocupado, el frontend
    no manda el siguiente frame hasta recibir respuesta (evita acumular cola).

Instalación:
  pip install -r requirements.txt

Ejecutar:
  python app.py
  -> abre http://localhost:5000
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

# Ruta absoluta basada en la ubicación de este archivo, para que funcione sin
# importar la estructura del proyecto (backend/app.py + frontend/, o todo en raíz).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
if not os.path.isdir(FRONTEND_DIR):
    # Estructura backend/app.py junto a ../frontend
    FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(__name__, template_folder=FRONTEND_DIR, static_folder=FRONTEND_DIR)
app.config["SECRET_KEY"] = "deepface-realtime-demo"

# threading: usamos eventlet/gevent en producción; threading simple sirve para esta demo
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading", max_http_buffer_size=10_000_000)

# Pool de hilos para no bloquear el loop de SocketIO mientras DeepFace corre (es CPU-bound).
executor = ThreadPoolExecutor(max_workers=2)

# Acciones que pedimos a DeepFace. Quitar alguna acelera bastante (emotion es la más liviana).
ACTIONS = ["emotion", "age", "gender", "race"]

# Detector backend: 'opencv' es el más rápido (peor recall). 'retinaface'/'mtcnn' son más
# precisos pero mucho más lentos -> en tiempo real conviene 'opencv' o 'ssd'.
DETECTOR_BACKEND = "opencv"

# Evita reprocesar si ya hay un análisis en curso (back-pressure simple por cliente)
busy_clients = set()


def decode_frame(data_url: str) -> np.ndarray:
    """Convierte un data URL 'data:image/jpeg;base64,...' a imagen OpenCV (BGR)."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_bytes, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


def analyze_frame(frame: np.ndarray):
    """Corre DeepFace sobre un frame. Devuelve lista de resultados (uno por cara) o []."""
    try:
        results = DeepFace.analyze(
            img_path=frame,
            actions=ACTIONS,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,  # no lanzar excepción si no hay cara
            silent=True,
        )
        # DeepFace devuelve un dict si hay 1 cara o una lista si hay varias
        if isinstance(results, dict):
            results = [results]
        return results
    except Exception as e:
        return {"error": str(e)}


@app.route("/")
def index():
    return render_template("index.html")


@socketio.on("connect")
def on_connect():
    print("Cliente conectado")


@socketio.on("disconnect")
def on_disconnect():
    busy_clients.discard(_request_sid())


def _request_sid():
    from flask import request
    return request.sid


def to_native(obj):
    """
    Convierte recursivamente tipos numpy (float32, int64, ndarray, etc.) a tipos
    nativos de Python. DeepFace devuelve números en tipos numpy que NO son
    serializables a JSON por defecto -> sin esto, socketio.emit() falla en
    silencio dentro del hilo y el frontend nunca recibe el resultado.
    """
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


@socketio.on("frame")
def on_frame(data):
    """
    Recibe un frame del frontend.
    `data` = {"image": "data:image/jpeg;base64,..."}
    """
    from flask import request
    sid = request.sid
    print(f"[DEBUG] Frame recibido de {sid}")  # <-- debug temporal

    # Back-pressure: si ya estamos procesando un frame de este cliente, lo ignoramos.
    if sid in busy_clients:
        print("[DEBUG] Cliente ocupado, frame ignorado")  # <-- debug temporal
        return
    busy_clients.add(sid)

    t0 = time.time()

    try:
        frame = decode_frame(data["image"])
        print(f"[DEBUG] Frame decodificado, shape={frame.shape}")  # <-- debug temporal
    except Exception as e:
        print(f"[DEBUG] Error decodificando: {e}")  # <-- debug temporal
        socketio.emit("result", {"error": f"No se pudo decodificar el frame: {e}"}, to=sid)
        busy_clients.discard(sid)
        return

    def task():
        try:
            print("[DEBUG] Iniciando análisis DeepFace...")  # <-- debug temporal
            result = analyze_frame(frame)
            result = to_native(result)  # <-- conversión crítica para que el JSON no falle
            print(f"[DEBUG] Análisis terminado: {result if isinstance(result, dict) else 'ok, ' + str(len(result)) + ' cara(s)'}")  # <-- debug temporal
            elapsed_ms = round((time.time() - t0) * 1000, 1)
            socketio.emit("result", {"faces": result, "latency_ms": elapsed_ms}, to=sid)
        except Exception as e:
            print(f"[DEBUG] ERROR en el hilo de análisis: {e}")  # <-- debug temporal
            socketio.emit("result", {"error": str(e)}, to=sid)
        finally:
            busy_clients.discard(sid)

    executor.submit(task)


if __name__ == "__main__":
    print("Cargando modelos de DeepFace (puede tardar la primera vez)...")

    # Precalentar modelos
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)

    try:
        analyze_frame(dummy)
    except Exception as e:
        print(f"Error precalentando DeepFace: {e}")

    print("Servidor iniciado")

    socketio.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        allow_unsafe_werkzeug=True
    )