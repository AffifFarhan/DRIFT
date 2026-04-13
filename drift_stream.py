import cv2
from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from starlette.background import BackgroundTask
import io

app = FastAPI()

# Global shared camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("? Could not open camera on startup.")

# MJPEG Streaming generator
def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

@app.get("/image")
def get_image():
    ret, frame = cap.read()
    if not ret:
        return Response(status_code=500)

    _, buffer = cv2.imencode('.jpg', frame)
    return Response(content=buffer.tobytes(), media_type="image/jpeg")

@app.get("/mjpeg")
def mjpeg():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.on_event("shutdown")
def shutdown_event():
    print("?? Releasing camera.")
    cap.release()
