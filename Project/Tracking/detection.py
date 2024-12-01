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
    cap = cv2.VideoCapture("../Reflection_5.mp4")
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden.")
        return

    # Lade die Haar-Cascade-Dateien für Full Body, Upper Body und Profilgesichter
    full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    bgs_handler = BackgroundSubtractionHandler()
    bgs_handler.set_subtractor("MOG2")

    all_person_features = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_counter % 2  == 0:


            # Hintergrundsubtraktion
            fg_frame = bgs_handler.apply_subtractor(frame)
            fg_frame = bgs_handler.post_processing(fg_frame)

            # Konturen finden
            contours, _ = cv2.findContours(fg_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                if cv2.contourArea(contour) > 12000:  # Filtere kleine Konturen aus
                    # Berechne das BoundingBox der Konturen
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


                    # Führe Full Body Detection durch
                    full_bodies = full_body_cascade.detectMultiScale(
                        fg_frame, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                    )

                    # Führe Upper Body Detection durch
                    upper_bodies = upper_body_cascade.detectMultiScale(
                        fg_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                    )

                    # Führe Profilgesichtserkennung durch
                    profile_faces = profile_face_cascade.detectMultiScale(
                        fg_frame, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30)
                    )

                    # Wenn Full Body, Upper Body oder Profilgesicht erkannt wird, markiere die Region
                    full_detected = False
                    upper_detected = False
                    profileface_detected = False

                    if full_bodies is not None:
                        full_detected = True
                    if upper_bodies is not None:
                        upper_detected = True
                    if profile_faces is not None:
                        profileface_detected = True

                    # Zähle positive Erkennungen
                    detection_count = sum([full_detected, upper_detected, profileface_detected])

                    # Wenn mindestens zwei Erkennungen zutreffen, markiere den Bereich als "Person erkannt"
                    if detection_count >= 2:
                        cv2.putText(frame, "Person erkannt", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                        # ORB-Features extrahieren
                        orb_features = extract_orb_features(frame, (x, y, w, h))
                        all_person_features.append(orb_features)

                        # Visualisiere die Keypoints
                        roi = frame[y:y + h, x:x + w]
                        roi_with_keypoints = cv2.drawKeypoints(roi, orb_features["keypoints"], None, color=(0, 255, 0))
                        frame[y:y + h, x:x + w] = roi_with_keypoints
        frame_counter += 1

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
