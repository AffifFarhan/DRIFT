import os
import cv2
import csv
import time
import threading
from datetime import datetime
from ultralytics import YOLO
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import psutil

MODEL_PATH = "/home/hikaru/Desktop/DRIFT/runs/detect/train2/weights/best.pt"
LOG_DIR = "/home/hikaru/Desktop/DRIFT/logs"
CAMERA_INDEX = 0
CONFIDENCE = 0.5
IMG_SIZE = 416

ALERT_CLASSES = ["close_eyeL", "close_eyeR", "yawn"]
DISPLAY_NAMES = {
    "close_eyeL": "Left Eye Closed",
    "close_eyeR": "Right Eye Closed",
    "open_eyeL": "Left Eye Open",
    "open_eyeR": "Right Eye Open",
    "yawn": "Yawning",
    "no_yawn": "No Yawn"
}

os.makedirs(LOG_DIR, exist_ok=True)
model = YOLO(MODEL_PATH)
print("[INFO] Model loaded. Classes:", model.names)

app = FastAPI()
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

latest_frame = None
latest_events = []
current_fps = 0
frame_lock = threading.Lock()

def get_log_path():
    date_str = datetime.now().strftime("%d%m%y")
    return os.path.join(LOG_DIR, f"{date_str}_detection_log.csv")

def log_event(detection_type, confidence, alert_triggered):
    path = get_log_path()
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp", "detection_type", "confidence", "alert_triggered"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            detection_type,
            round(confidence, 2),
            alert_triggered
        ])

def detection_loop():
    global latest_frame, latest_events, current_fps
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Detection loop started.")
    frame_count = 0
    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        results = model(frame.copy(), imgsz=IMG_SIZE, conf=CONFIDENCE, verbose=False)[0]
        annotated = results.plot()

        frame_count += 1
        elapsed = time.time() - prev_time
        if elapsed >= 1.0:
            current_fps = frame_count / elapsed
            prev_time = time.time()
            frame_count = 0

        cv2.putText(annotated, f"FPS: {current_fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = float(box.conf[0])
            alert = "yes" if name in ALERT_CLASSES else "no"
            log_event(name, conf, alert)

            event = {
                "time": datetime.now().strftime("%H:%M:%S"),
                "detection": DISPLAY_NAMES.get(name, name),
                "raw": name,
                "confidence": f"{conf:.0%}",
                "alert": alert
            }
            latest_events.insert(0, event)
            latest_events = latest_events[:50]

        with frame_lock:
            latest_frame = annotated.copy()

def generate_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
               buffer.tobytes() + b"\r\n")
        time.sleep(0.03)

@app.get("/mjpeg")
def mjpeg():
    return StreamingResponse(generate_frames(),
                             media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/events")
def events():
    return JSONResponse(latest_events[:20])

@app.get("/health")
def health():
    temp = None
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            temp = round(int(f.read()) / 1000, 1)
    except:
        pass
    return JSONResponse({
        "fps": round(current_fps, 1),
        "cpu": psutil.cpu_percent(),
        "ram_used": round(psutil.virtual_memory().used / 1024 / 1024 / 1024, 2),
        "ram_total": round(psutil.virtual_memory().total / 1024 / 1024 / 1024, 2),
        "temp": temp,
        "uptime": int(time.time() - psutil.boot_time())
    })

if __name__ == "__main__":
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
