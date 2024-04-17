import cv2
import numpy as np
import onnxruntime as rt

sess = rt.InferenceSession('yolov5/runs/train/exp/weights/best.onnx')

# Impostazioni di rilevamento
confidence_threshold = 0.7  # Soglia di confidenza aggiornata

# Funzione per trovare la posizione del box con la probabilità più alta e disegnare un cerchio in quella posizione
def draw_highest_probability_circle(frame, detections, frame_width, frame_height):
    max_confidence = 0
    max_confidence_position = None

    for detection in detections:
        for obj in detection:
            # Estrai il valore di confidenza
            confidence = obj[4]

            # Verifica se la confidenza supera la soglia e se è maggiore della massima finora
            if confidence > confidence_threshold and confidence > max_confidence:
                max_confidence = confidence
                startX_rel, startY_rel, _, _ = obj[0:4]

                # Converti le coordinate relative alle dimensioni del blob alle dimensioni effettive del frame
                startX = int(startX_rel * frame_width / 640)
                startY = int(startY_rel * frame_height / 480)

                max_confidence_position = (startX, startY)

    if max_confidence_position is not None:
        cv2.circle(frame, max_confidence_position, 15, (255, 255, 255), -1)

# Funzione per rilevare gli oggetti in un'immagine dalla webcam
def detect_objects_webcam():
    # Attiva la webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Leggi il frame dalla webcam
        ret, frame = cap.read()
        if not ret:
            break

        # Ripristina le dimensioni effettive del frame
        frame_height, frame_width = frame.shape[:2]

        # Prepara il frame per il rilevamento
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 480), swapRB=True, crop=False)

        # Esegui il rilevamento
        detections = sess.run(None, {'images': blob})[0]

        # Disegna un cerchio nella posizione del box con la probabilità più alta
        draw_highest_probability_circle(frame, detections, frame_width, frame_height)

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
