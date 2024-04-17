import cv2
import numpy as np
import onnxruntime as rt

sess = rt.InferenceSession('yolov5/runs/train/exp/weights/best.onnx')

# Impostazioni di rilevamento
confidence_threshold = 0.6 # Soglia di confidenza aggiornata

# Funzione per trovare la posizione del box con la probabilità più alta e disegnare un rettangolo in quella posizione
def draw_highest_probability_rectangle(frame, detections, frame_width, frame_height):
    for detection in detections:
        max_confidence = 0
        
        # Trova la probabilità più alta tra tutti gli oggetti rilevati
        for obj in detection:
            confidence = obj[4]
            if confidence > max_confidence:
                max_confidence = confidence

        # Disegna un rettangolo solo per gli oggetti con la probabilità più alta
        for obj in detection:
            confidence = obj[4]
            if confidence == max_confidence and confidence > confidence_threshold:
                centerX = int(obj[0] * frame_width /640)
                centerY = int(obj[1] * frame_height /480)
                width = int(obj[2] * frame_width /640)
                height = int(obj[3] * frame_height /480)

                startX = int(centerX - width / 2)
                startY = int(centerY - height / 2)

                endX = startX + width
                endY = startY + height

                cv2.rectangle(frame, (startX, startY), (endX, endY), (255, 255, 255), 2)

# Funzione per rilevare gli oggetti in un'immagine dalla webcam
def detect_objects_webcam():
    # Attiva la webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        # Leggi il frame dalla webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Ripristina le dimensioni effettive del frame
        frame_height, frame_width = frame.shape[:2]

        # Prepara il frame per il rilevamento
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (640, 480)), 1/255.0, (640, 480), swapRB=True, crop=False)

        # Esegui il rilevamento
        detections = sess.run(None, {'images': blob})[0]

        # Disegna un rettangolo intorno agli oggetti rilevati con alta probabilità
        draw_highest_probability_rectangle(frame, detections, frame_width, frame_height)

        # Mostra il frame con i rilevamenti
        cv2.imshow('Rilevamenti', frame)

        # Interrompi l'esecuzione se viene premuto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia la risorsa della webcam e chiudi tutte le finestre
    cap.release()
    cv2.destroyAllWindows()

# Esegui la funzione per il rilevamento degli oggetti tramite la webcam
detect_objects_webcam()
