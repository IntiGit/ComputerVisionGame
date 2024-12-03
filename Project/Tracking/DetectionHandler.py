import cv2
import numpy as np
from Project.Tracking.backgroundSubtractionHandler import BackgroundSubtractionHandler


def calculate_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity


def detect_head_and_face(frame, bbox, bgs_frame, face_cascade, profil_face_cascade):
    x, y, w, h = bbox

    head_mask = bgs_frame[y:y + int(h * 0.15), x:x + w]
    _, head_mask = cv2.threshold(head_mask, 100, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    circles = []

    for contour in contours:
        circularity = calculate_circularity(contour)
        if circularity > 0.3:
            circles.append(contour)

    face_detected = False
    profile_face_detected = False
    circle_detected = False

    if circles:
        best_circle = max(circles, key=calculate_circularity)
        (x_center, y_center), radius = cv2.minEnclosingCircle(best_circle)
        x_center, y_center, radius = int(x_center), int(y_center), int(radius)

        # Verschiebe den Kreis relativ zur Bounding Box
        x_center += x
        y_center += y
        circle_detected = True
        face_roi = frame[max(0, y_center - radius):min(frame.shape[0], y_center + radius),
                   max(0, x_center - radius):min(frame.shape[1], x_center + radius)]

        # Gesichter erkennen
        faces = face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=1, minSize=(30, 30))
        profil_faces = profil_face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=1, minSize=(10, 10))

        for (fx, fy, fw, fh) in profil_faces:
            profile_face_detected = True

        for (fx, fy, fw, fh) in faces:
            face_detected = True

        if circle_detected and (face_detected or profile_face_detected):
            cv2.putText(frame, "Person erkannt", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), 2)
            cv2.putText(frame, "Gesicht erkannt", (x_center - radius, y_center - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.rectangle(frame, (x_center - radius, y_center - radius),
                          (x_center - radius, y_center - radius), (0, 255, 0), 2)

    return circle_detected and (face_detected or profile_face_detected)


def detect_body(frame, bounding_box, fg_frame, body_cascade):
    x, y, w, h = bounding_box
    roi = fg_frame[y:y + h, x:x + w]
    _, head_mask = cv2.threshold(roi, 100, 255, cv2.THRESH_BINARY)

    bodies = body_cascade.detectMultiScale(
        roi,
        scaleFactor=1.1,
        minNeighbors=2,
        minSize=(10, 10),
    )
    if len(bodies) > 0:
        cv2.putText(frame, "Person erkannt", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        return True  # Oberkörper wurde erkannt
    return False


def person_detection():
    cap = cv2.VideoCapture("../Brick_3.mp4")
    if not cap.isOpened():
        print("Video konnte nicht geöffnet werden.")
        return

    # Lade die Haar-Cascade-Dateien für Full Body und Gesicht
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_upperbody.xml")
    profile_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
    frontal_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    prev_moments = None
    frame_counter = 0  # Zähler für Frames

    # Background Subtraction Handler
    bgs_handler = BackgroundSubtractionHandler()
    bgs_handler.set_subtractor("MOG2")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Hintergrundsubtraktion
        fg_frame = bgs_handler.apply_subtractor(frame)
        fg_frame = bgs_handler.post_processing(fg_frame)

        # Begrenze das Bild auf die mittleren 70% (ignoriere obere 20% und untere 10%)
        height, width = fg_frame.shape[:2]
        top_limit = int(height * 0.2)
        bottom_limit = int(height * 0.9)
        fg_frame[:top_limit, :] = 0  # Obere 20%
        fg_frame[bottom_limit:, :] = 0  # Untere 10%

        contours, _ = cv2.findContours(fg_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)
            if area < 12000:
                continue
            aspect_ratio = w / h
            if not (0.3 <= aspect_ratio <= 0.8):  # Breite-Höhe-Verhältnis prüfen
                continue
            if detect_head_and_face(frame, (x, y, w, h), fg_frame, frontal_face_cascade,
                                    profile_face_cascade):
                continue
            detect_body(frame, (x, y, w, h), fg_frame, upper_body_cascade)

        # Frame anzeigen
        cv2.imshow("Personen Detektion", frame)
        cv2.imshow("BGS", fg_frame)

        # Abbruch bei "q"
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


person_detection()
