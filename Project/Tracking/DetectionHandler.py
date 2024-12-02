import cv2
import numpy as np
from Project.Tracking.backgroundSubtractionHandler import BackgroundSubtractionHandler


# Farbhistogramm-Funktion
def calc_color_histogram(roi):
    # Berechnung des Farbhistogramms
    hist = cv2.calcHist([roi], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def calculate_momentum(prev_moments, current_moments):
    # Berechnung des Momentums basierend auf den Schwerpunkten
    if prev_moments is not None:
        # Berechnung des Schwerpunkts der vorherigen und aktuellen Kontur
        prev_center = (int(prev_moments['m10'] / prev_moments['m00']), int(prev_moments['m01'] / prev_moments['m00']))
        current_center = (
        int(current_moments['m10'] / current_moments['m00']), int(current_moments['m01'] / current_moments['m00']))

        # Berechnung der Verschiebung des Schwerpunkts
        dx = current_center[0] - prev_center[0]
        dy = current_center[1] - prev_center[1]

        # Berechnung des Momentums (Betrag der Verschiebung)
        momentum = np.sqrt(dx ** 2 + dy ** 2)
        return momentum
    return 0


# HOG-basierte Merkmalsberechnung (ohne HOG Descriptor)
def compute_hog_features(roi):
    # Berechne HOG über den Gradienten (manuelle Berechnung)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)

    # Unterteile das Bild in Zellen und berechne HOG-Manuell (für jedes Zellen-Array)
    cell_size = 8  # z.B. 8x8 Pixel pro Zelle
    cells_per_block = 2  # z.B. 2x2 Zellen pro Block

    num_cells_x = gray.shape[1] // cell_size
    num_cells_y = gray.shape[0] // cell_size
    hog_features = []

    for y in range(num_cells_y):
        for x in range(num_cells_x):
            # Extrahiere Histogramm für jede Zelle
            cell_mag = mag[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]
            cell_angle = angle[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size]

            # Berechnung von Histogrammen für die Zelle
            hist, _ = np.histogram(cell_angle, bins=9, range=(0, 180), weights=cell_mag)
            hog_features.extend(hist)

    return hog_features


# ORB-Feature-Extraktion
def extract_orb_features(frame, bbox):
    x, y, w, h = bbox
    roi = frame[y:y + h, x:x + w]

    # ORB initialisieren
    orb = cv2.ORB_create()

    # Graustufenbild für ORB
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Keypoints und Deskriptoren extrahieren
    keypoints, descriptors = orb.detectAndCompute(gray_roi, None)
    return {"keypoints": keypoints, "descriptors": descriptors}


def calculate_circularity(contour):
    """
    Berechnet die Circularität eines Objekts anhand der Formel C = (4 * pi * Area) / (Perimeter^2).

    :param contour: Der Kontur des Objekts.
    :return: Der Circularitätswert des Objekts.
    """
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return 0

    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity


def detect_head_and_face(frame, bbox, bgs_frame, face_cascade, profil_face_cascade, upper_body_cascade):
    """
    Funktion zur Bestimmung des Kopfbereichs (mit der höchsten Circularität) und anschließender Gesichtserkennung
    im oberen Bereich der Bounding Box.

    :param frame: Das aktuelle Bild
    :param bbox: Bounding Box (x, y, w, h)
    :param bgs_frame: Die Hintergrundsubtraktionsmaske
    :param face_cascade: Der Haar-Cascade-Classifier für Gesichter
    :param profil_face_cascade: Der Haar-Cascade-Classifier für Profilgesichter
    :param upper_body_cascade: Der Haar-Cascade-Classifier für Oberkörper
    :return: Das Bild mit markiertem Kopfbereich und Gesicht
    """
    x, y, w, h = bbox

    # Berechne den neuen Höhe-Wert für die oberen 60 % der Bounding Box
    upper_body_height = int(h * 0.5)

    # Maskiere den Kopfbereich im BGS Bild (oberer 10% der Bounding Box)
    head_mask = bgs_frame[y:y + int(h * 0.1), x:x + w]

    body_roi = frame[y + int(h * 0.3):y + upper_body_height, x:x + w]

    # Finde die Konturen
    contours, _ = cv2.findContours(head_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Liste für Kreise mit hoher Circularität
    circles = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 300:  # Ignoriere kleine Konturen
            continue

        # Berechne die Circularität des Objekts
        circularity = calculate_circularity(contour)

        # Wenn die Circularität einen hohen Wert hat, speichern wir es als potenziellen Kopf
        if circularity > 0.3:  # Ein gewisser Schwellenwert für Circularität
            circles.append(contour)

    face_detected = False  # Flag für die Gesichtserkennung
    profile_face_detected = False  # Flag für die Profil-Gesichtserkennung
    upper_boddy_detected = False  # Flag für die Oberkörper-Erkennung

    # Wenn wir mindestens einen Kreis gefunden haben, wähle den mit der höchsten Circularität
    if circles:
        best_circle = max(circles, key=calculate_circularity)

        # Berechne das Zentrum und den Radius des besten Kreises
        (x_center, y_center), radius = cv2.minEnclosingCircle(best_circle)

        # Umwandeln der Koordinaten und Radius in Ganzzahlen
        x_center, y_center, radius = int(x_center), int(y_center), int(radius)

        # Die Koordinaten des Kreises in Bezug auf die Bounding Box anpassen
        x_center += x  # Verschiebe den Kreis relativ zur Bounding Box
        y_center += y  # Verschiebe den Kreis relativ zur Bounding Box

        # Setze das Flag für die Kreis-Erkennung
        circle_detected = True

        # Gesichtserkennung im Bereich des Kopfes (innerhalb des Kreises)
        face_roi = frame[max(0, y_center - radius):min(frame.shape[0], y_center + radius),
                   max(0, x_center - radius):min(frame.shape[1], x_center + radius)]

        # Gesichter erkennen
        faces = face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        profil_faces = profil_face_cascade.detectMultiScale(face_roi, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))
        upper_boddies = upper_body_cascade.detectMultiScale(body_roi, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

        # Erkenne Oberkörper und markiere sie
        for (fx, fy, fw, fh) in upper_boddies:
            upper_boddy_detected = True

        # Erkenne Profilgesichter
        for (fx, fy, fw, fh) in profil_faces:
            profile_face_detected = True

        # Erkenne normale Gesichter
        for (fx, fy, fw, fh) in faces:
            face_detected = True

        # Nur wenn sowohl Kreis als auch Gesicht oder Oberkörper erkannt wurden, wird die Nachricht angezeigt
        if circle_detected and ( (face_detected or profile_face_detected) or upper_boddy_detected ):
            cv2.putText(frame, "Person erkannt", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Zeichne den Kreis auf das Bild
            cv2.circle(frame, (x_center, y_center), radius, (0, 255, 0), 2)
            cv2.rectangle(frame, (x_center - radius, y_center - radius),
                          (x_center - radius, y_center - radius), (0, 255, 0), 2)
            cv2.putText(frame, "Gesicht erkannt", (x_center - radius, y_center - radius - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


def person_detection():
    cap = cv2.VideoCapture("../autos.mov")
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

        # Begrenze das Bild auf die unteren 90%
        height, width = fg_frame.shape[:2]
        roi_height = int(height * 0.8)  # Berechne 90% der Höhe
        fg_frame[:height - roi_height, :] = 0  # Setze die oberen 10% des Bildes auf 0

        contours, _ = cv2.findContours(fg_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, w, h = cv2.boundingRect(contour)

            if area < 12000:
                continue
            if w > 200:
                continue
            aspect_ratio = w / h
            if 0.3 <= aspect_ratio <= 0.6:
                detect_head_and_face(frame, (x, y, w, h), fg_frame, frontal_face_cascade,
                                     profile_face_cascade,upper_body_cascade)



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
