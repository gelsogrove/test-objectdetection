import cv2
from ultralytics import YOLO

model_path = "models/cripta/cripta.onnx"

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


model = YOLO(model_path, task="detect", verbose=False)


while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        if frame.size > 0:
            results = model(frame, verbose=False)
            boxes = results[0].boxes.xyxy.tolist()
            classes = results[0].boxes.cls.tolist()
            names = results[0].names
            confidences = results[0].boxes.conf.tolist()

            for box, cls, conf in zip(boxes, classes, confidences):
                confidence_threshold = 0.7  # Set your desired confidence threshold here
                if conf > confidence_threshold:
                    x1, y1, x2, y2 = map(int, box)
                    confidence = conf
                    detected_class = cls
                    name = names[int(cls)]
                    print(f"Object Detected: {name}")
        

# Release the capture
cap.release()
cv2.destroyAllWindows()
