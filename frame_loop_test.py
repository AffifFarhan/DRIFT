import os
os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("? Failed to open camera")
    exit()

print("?? Starting OpenCV frame loop...")
fps = 0
frame_count = 0
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("?? Failed to grab frame")
        continue

    frame_count += 1
    elapsed = time.time() - prev_time
    if elapsed >= 1.0:
        fps = frame_count / elapsed
        print(f"FPS: {fps:.2f}")
        prev_time = time.time()
        frame_count = 0

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Test Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("?? Quit requested")
        break

cap.release()
cv2.destroyAllWindows()
