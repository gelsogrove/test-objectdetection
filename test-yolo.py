#!/usr/bin/env python3
import cv2
import math 
import torch
import torchvision.transforms as transforms
from PIL import Image
from ultralytics import YOLO

# Avvia la webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Modello YOLO
model_yolo = YOLO("./best.pt")

# Modello per la segmentazione
model_segmentation = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet101', pretrained=True)
model_segmentation.eval()

# Le tue etichette personalizzate
custom_labels = ["Arcangelo-Gabriele", "Crocefissione", "San-Carlo", "ultima-cena"]  

while True:
    success, img = cap.read()
    results = model_yolo(img, stream=True)

    # Coordinate
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # Bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # converti in valori interi

            # Confidence
            confidence = math.ceil((box.conf[0] * 100)) / 100

            if confidence > 0.8:  # Aggiungi l'if per controllare la confidence
                # Metti il box nella webcam
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # Nome della classe©©
                cls = int(box.cls[0])
                class_name = custom_labels[cls]

                # Dettagli dell'oggetto
                org = (x1, y1 - 10)  # Posiziona il testo sopra il box
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                color = (255, 0, 0)
                thickness = 1

                text = f"{class_name} ({confidence:.2f})"
                cv2.putText(img, text, org, font, fontScale, color, thickness)

    # Effettua la segmentazione dell'immagine
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    preprocess = transforms.Compose([
        transforms.Resize((512, 512)),  # Aggiusta le dimensioni dell'input del modello di segmentazione
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model_segmentation.to('cuda')
    with torch.no_grad():
        output = model_segmentation(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # Visualizza la segmentazione
    segmentation_img = output_predictions.byte().cpu().numpy()  # Converti l'output in un'immagine
    cv2.imshow('Segmentation', segmentation_img)

    # Visualizza l'immagine con i rilevamenti
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
