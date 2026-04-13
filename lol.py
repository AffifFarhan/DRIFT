import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time
from ultralytics import YOLO

print("? Loading model...")
model = YOLO("/home/hikaru/Desktop/DRIFT/runs/detect/train2/weights/best.pt")
print("? Model loaded. Classes:", model.names)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("? Failed to open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

prev_time = time.time()
frame_count = 0

print("?? Starting detection loop...")
while True:
    ret, frame = cap.read()
    if not ret:
        print("?? Failed to read frame. Retrying...")
        continue

    # Predict and annotate
    results = model.predict(source=frame, conf=0.6, imgsz=416, verbose=False)[0]
    annotated = results.plot()

    # Print detected class names
    detected_classes = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        class_name = model.names[cls_id]
        detected_classes.append(class_name)

    print("Detected:", detected_classes)

    # FPS
    frame_count += 1
    curr_time = time.time()
    elapsed = curr_time - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        prev_time = curr_time
        frame_count = 0

    # Draw FPS
    cv2.putText(annotated, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Display
    cv2.imshow("DRIFT Detection", annotated)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("?? Quitting stream.")
        break

cap.release()
cv2.destroyAllWindows()
print("? Exited cleanly.")
