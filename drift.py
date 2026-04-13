import os
import cv2
import time
import torch
from ultralytics import YOLO

# Fix for Qt error
os.environ["QT_QPA_PLATFORM"] = "xcb"

# Load model once
model = YOLO("/home/hikaru/Desktop/DRIFT/runs/detect/train2/weights/best.pt")
print("? YOLOv8 model loaded:", model.names)

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("? Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# FPS counter
prev_time = time.time()
frame_count = 0
fps = 0

print("?? Starting YOLOv8 Detection Loop...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("?? No frame captured.")
            continue

        # Avoid in-place modification
        input_frame = frame.copy()

        # Run inference (more stable syntax)
        results_list = model(input_frame, imgsz=416, conf=0.5, verbose=False)
        results = results_list[0]  # Single frame

        # Draw results and extract detections
        annotated = results.plot()
        class_names = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            name = model.names.get(cls_id, f"id_{cls_id}")
            class_names.append(name)

        print("?? Detected:", class_names)

        # FPS calculation
        frame_count += 1
        elapsed = time.time() - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            prev_time = time.time()
            frame_count = 0

        cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show window
        cv2.imshow("?? DRIFT Detection", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("?? Exit requested.")
            break

except KeyboardInterrupt:
    print("?? Interrupted manually.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("? Shutdown complete.")
