from Project.Tracking.backgroundSubtractionHandler import BackgroundSubtractionHandler
import cv2


class DetectionHandler:
    def __init__(self):
        self.bgs_handler = BackgroundSubtractionHandler()
        self.bgs_handler.set_subtractor("MOG2")
        # Lade die Haar-Cascade-Dateien für Full Body, Upper Body und Profilgesichter
        self.full_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")
        self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
        self.profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

    def __call__(self, frame):
        fg_frame = self.bgs_handler.apply_subtractor(frame)
        fg_frame = self.bgs_handler.post_processing(fg_frame)

        # Konturen finden
        contours, _ = cv2.findContours(fg_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 12000:  # Filtere kleine Konturen aus
                x, y, w, h = cv2.boundingRect(contour)

                # Führe Full Body Detection durch
                full_bodies = self.full_body_cascade.detectMultiScale(
                    fg_frame, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30)
                )

                # Führe Upper Body Detection durch
                upper_bodies = self.upper_body_cascade.detectMultiScale(
                    fg_frame, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                )

                # Führe Profilgesichtserkennung durch
                profile_faces = self.profile_face_cascade.detectMultiScale(
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
                    positions.append((x + w // 2, y + h // 2))  # Zentrum der Bounding Box
        return fg_frame, positions
