import cv2
import numpy as np

# Kreise finden
def findCircles(boxes, frame):
    best_box = None
    min_distance = float('inf')

    for (x, y, w, h) in boxes:
        # Erweiterten Suchbereich oberhalb und innerhalb der Box
        head_region_x = max(x - 20, 0)
        head_region_y = max(y - h // 4, 0)
        head_region_w = w + 40
        head_region_h = int(h * 0.5)

        roi = frame[head_region_y:head_region_y + head_region_h, head_region_x:head_region_x + head_region_w]

        gray_roi = cv2.medianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 5)

        # Hough-Transformation
        circles = cv2.HoughCircles(
            gray_roi,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=20,
            maxRadius=40
        )
        # Wenn Kreise gefunden wurden, berechne die Distanz zum Box-Zentrum
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for circle in circles:
                cx, cy, radius = circle
                circle_center = np.int32((head_region_x + cx, head_region_y + cy))
                box_top_center = (x + w // 2, y)
                distance = np.sqrt(
                    (circle_center[0] - box_top_center[0]) ** 2 + (circle_center[1] - box_top_center[1]) ** 2)

                if distance < min_distance:
                    min_distance = distance
                    best_box = (x, y, w, h)

            return best_box
    return None

# Zusammenführen von überlappenden Boxen
def merge_overlapping_boxes(boxes):
    if len(boxes) == 0:
        return []

    # Sortiere Boxen nach X-Koordinate
    boxes = sorted(boxes, key=lambda x: x[0])
    merged_boxes = []

    # Mergen
    current_box = boxes[0]
    for i in range(1, len(boxes)):
        x, y, w, h = current_box
        nx, ny, nw, nh = boxes[i]

        # Berechne Überlappungsbereich
        if nx <= x + w and ny <= y + h:
            merged_x = min(x, nx)
            merged_y = min(y, ny)
            merged_w = max(x + w, nx + nw) - merged_x
            merged_h = max(y + h, ny + nh) - merged_y
            current_box = (merged_x, merged_y, merged_w, merged_h)
        else:
            merged_boxes.append(current_box)
            current_box = boxes[i]

    merged_boxes.append(current_box)
    return merged_boxes

# Box mit der höchsten varianz erhalten
def getBox_mostVariance(boxes, frame):
    max_normalized_variance = -1
    most_uneven_box = None

    for (x, y, w, h) in boxes:
        roi = frame[y:y + h, x:x + w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        variance = np.var(gray_roi)
        num_pixels = gray_roi.size
        normalized_variance = variance / num_pixels

        if normalized_variance > max_normalized_variance:
            max_normalized_variance = normalized_variance
            most_uneven_box = (x, y, w, h)

    return most_uneven_box


# Boxen filtern nach Größe und Seitenverhältnis
def selectCandidates(contours):
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 5000 or area > 85000:  # Filter für Konturfläche
            continue

        (x, y, w, h) = cv2.boundingRect(contour)

        ratio = h / w

        if not (1.5 < ratio < 4.0):  # Filter für Seitenverhältnis
            continue
        candidates.append((x, y, w, h))
    return candidates

# Extrahieren von ORB-Features aus einer Box
def extract_orb_features(image, bounding_box):
    x, y, w, h = bounding_box
    cropped = image[y:y + h, x:x + w]

    orb = cv2.ORB_create(nfeatures=500)
    keypoints, descriptors = orb.detectAndCompute(cropped, None)

    return keypoints, descriptors


# Anzahl an Übereinstimmungen zwischen 2 ORB-Deskriptoren finden
def match_orb_features(descriptors1, descriptors2):
    if descriptors1 is None or descriptors2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2) # Vergleichen von ORB-Feature
    return len(matches)

# beste Box basierend auf ORB-Features auswählen
def select_consistent_box_orb(image, bounding_boxes, reference_descriptor):
    max_matches = 0
    best_box = None

    for i, box in enumerate(bounding_boxes):
        _, descriptors = extract_orb_features(image, box)
        matches = match_orb_features(descriptors, reference_descriptor)
        if matches >= 50 and matches > max_matches:
            max_matches = matches
            best_box = box

    return best_box


# Person detektieren
def detectPerson(frame, subtractor, ref_descriptors, lastDetection):

    # BGS anwenden
    fgmask = subtractor.apply(frame)
    # Opening, Dilatation, Closing
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
    fgmask = cv2.dilate(fgmask, np.ones((9, 9)))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)))
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = selectCandidates(contours)

    found = len(candidates) != 0

    merged_candidates = merge_overlapping_boxes(candidates)

    # Wenn es schon einen ORB-DEskriptor für das Objekt gibt, die am besten
    # passende Box zurückgeben
    if ref_descriptors is not None:
        consistent_box = select_consistent_box_orb(frame, merged_candidates, ref_descriptors)
        if consistent_box is None and lastDetection is not None:
            for box in merged_candidates:
                x, y, w, h = box
                lx, ly, lw, lh = lastDetection
                if abs((w * h) - (lw*lh)) < 30000 and abs(x - lx) < 100:
                    return box

        return consistent_box
    # Wenn keine konsistente Box gefunden wurde, dann beste Box mit Kreisen in der Nähe oder Box mit höchster Varianz
    best_box = None
    if found:
        best_box = findCircles(merged_candidates, frame)

    result = None
    if best_box:
        result = best_box
    elif merged_candidates:
        result = getBox_mostVariance(merged_candidates, frame)

    return result


def showBGS(frame, sub):
    fgmask = sub.apply(frame)

    height, width = fgmask.shape[:2]
    bottom = int(height * 0.1)
    fgmask[-bottom:, :] = 0

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    fgmask = cv2.dilate(fgmask, np.ones((9, 9)))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

    cv2.imshow('BGS', fgmask)


def showDetection(frame, sub):
    boundingBox = detectPerson(frame, sub)
    if boundingBox is not None:
        x, y, w, h = boundingBox
        cv2.rectangle(frame, (x, y), (x + w, y + h + int(0.1 * h)), (0, 255, 0), 2)
    cv2.imshow('Detection', frame)
