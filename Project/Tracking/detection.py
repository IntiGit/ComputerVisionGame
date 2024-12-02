import cv2
import numpy as np
from Project.Tracking.backgroundSubtractionHandler import BackgroundSubtractionHandler

def extract_orb_features(frame, bbox):
    x, y, w, h = bbox
    roi = frame[y:y+h, x:x+w]  # Region of Interest (ROI)

    # Konvertiere in Graustufen für ORB
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # ORB initialisieren
    orb = cv2.ORB_create()

    # Keypoints und Deskriptoren berechnen
    keypoints, descriptors = orb.detectAndCompute(gray_roi, None)

    return {
        "keypoints": keypoints,
        "descriptors": descriptors
    }


def detection():
    # Counter für jeden 2. Frame
    frame_counter = 0
    # Video laden
    cap = cv2.VideoCapture("../autos.mov")
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden.")
        return

    # Lade die Haar-Cascade-Dateien für Full Body, Upper Body und Profilgesichter
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    bgs_handler = BackgroundSubtractionHandler()
    bgs_handler.set_subtractor("MOG2")
    tracked_positions = []  # Für Bewegungsverfolgung
    all_person_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break


        # Hintergrundsubtraktion
        fg_frame = bgs_handler.apply_subtractor(frame)
        fg_frame = bgs_handler.post_processing(fg_frame)

        # Begrenze das Bild auf die unteren 90%
        height, width = fg_frame.shape[:2]
        roi_height = int(height * 0.9)  # Berechne 90% der Höhe
        fg_frame[:height - roi_height, :] = 0  # Setze die oberen 10% des Bildes auf 0


        # Konturen finden
        contours, _ = cv2.findContours(fg_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 12000:  # Filtere kleine Konturen aus
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h  # Berechne Seitenverhältnis
            # Filter für personähnliche Konturen
            if (0.3 <= aspect_ratio <= 0.6) or (0.6 <= aspect_ratio <= 1.0):



                cv2.putText(frame, "Person erkannt", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # ORB-Features extrahieren
                orb_features = extract_orb_features(frame, (x, y, w, h))
                all_person_features.append(orb_features)

                # Visualisiere die Keypoints
                roi = frame[y:y + h, x:x + w]
                roi_with_keypoints = cv2.drawKeypoints(roi, orb_features["keypoints"], None,
                                                       color=(0, 255, 0))
                frame[y:y + h, x:x + w] = roi_with_keypoints

        # frame_counter += 1

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
