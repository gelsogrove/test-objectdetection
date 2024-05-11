import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('models/cripta/cripta.bt', verbose=False)

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Perform inference on the frame
    results = model(frame, verbose=False)

    # Extract bounding boxes, classes, names, and confidences
    boxes = results[0].boxes.xyxy.tolist()
    classes = results[0].boxes.cls.tolist()
    names = results[0].names
    confidences = results[0].boxes.conf.tolist()

    # Iterate through the results
    for box, cls, conf in zip(boxes, classes, confidences):
        confidence_threshold = 0.5  # Set your desired confidence threshold here
        if conf > confidence_threshold:
            x1, y1, x2, y2 = map(int, box)
            confidence = conf
            detected_class = cls
            name = names[int(cls)]

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)  # Purple color for the bounding box
            label = f'{name} {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)  # Larger text size
            
            # Print the name of the detected object to the console
            print(f"Object Detected: {name}")

    # Display the frame with bounding boxes
    cv2.imshow('YOLOv8 Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
