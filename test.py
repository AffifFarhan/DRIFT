from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/train2/weights/best.pt")  # Adjust path if different

# Initialize webcam (Razer Kiyo X usually on index 1)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

# Check if camera opened successfully
if not cap.isOpened():
    print("[ERROR] Cannot open webcam.")
    exit()

print("[INFO] Starting webcam detection... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Run inference
    results = model.predict(source=frame, imgsz=416, conf=0.4, verbose=False)[0]

    # Annotate results on the frame
    annotated_frame = results.plot()

    # Show the frame
    cv2.imshow("YOLOv8 Eye & Yawn Detection", annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
