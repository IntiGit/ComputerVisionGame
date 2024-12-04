import cv2
import numpy as np
from Project.Tracking.backgroundSubtractionHandler import BackgroundSubtractionHandler


def detection():
    # Video laden
    cap = cv2.VideoCapture("../Reflection_5.mp4")
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden.")
        return

    bgs_handler = BackgroundSubtractionHandler()
    bgs_handler.set_subtractor("MOG2")

    # Lade Haar Cascade für Gesichter
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Hintergrundsubtraktion
        fg_frame = bgs_handler.apply_subtractor(frame)
        fg_frame = bgs_handler.post_processing(fg_frame)

        # Gesichter im Vordergrund erkennen
        faces = face_cascade.detectMultiScale(fg_frame, scaleFactor=1.1, minNeighbors=2, minSize=(25, 25))
        if faces is not None:
            for (x, y, w, h) in faces:

                # Definiere den Körperbereich unterhalb des Kopfes
                body_x1 = max(0, x - w)  # Vergrößere Breite
                body_x2 = min(frame.shape[1], x + 2 * w)
                body_y1 = y + h
                body_y2 = min(frame.shape[0], y + 6 * h)  # Größere Höhe für Körper

                body_region = fg_frame[body_y1:body_y2, body_x1:body_x2]

                # Suche nach Konturen im Körperbereich
                contours, _ = cv2.findContours(body_region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    print(cv2.contourArea(contour))
                    if cv2.contourArea(contour) > 8000:  # Filtere kleinere Konturen aus
                        bx, by, bw, bh = cv2.boundingRect(contour)
                        # Zeichne das Rechteck um den Körper (Koordinaten in den Originalrahmen umrechnen)
                        cv2.rectangle(frame, (body_x1 + bx, body_y1 + by),
                                      (body_x1 + bx + bw, body_y1 + by + bh), (255, 0, 0), 2)
                        cv2.putText(frame, "Person erkannt", (body_x1 + bx, body_y1 + by - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Debug: Zeige die Hintergrundmaske
        cv2.imshow("Hintergrundmaske", fg_frame)

        # Ergebnis anzeigen
        cv2.imshow("Personen Detektion", frame)

        # Abbruch bei "q"
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


detection()
