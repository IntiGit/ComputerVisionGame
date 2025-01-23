import cv2
import numpy as np


# Zusammenführen von überlappenden Boxen
def merge_overlapping_boxes(boxes, reach=10):
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
        if nx <= x + w + reach and ny <= y + h:
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


def minimal_covering_rectangles(contour, frame, max_points_per_rect=50, min_height=50):
    # Konvertiere die Kontur zu einer flachen Liste von Punkten
    points = contour.reshape(-1, 2)

    points = points[np.argsort(points[:, 0])] # nach X-Koordinate sortieren

    rectangles = []

    # Zufällige Farbe für Debugging
    color = np.random.randint(0, 256, size=3)

    for i in range(0, len(points), max_points_per_rect):
        group = points[i:i + max_points_per_rect]

        x, y, w, h = cv2.boundingRect(group)

        if h >= min_height:
            rectangles.append((x, y, w, h))

    if len(rectangles) == 0: # keine Rechtecke gefunden
        return None

    # Rechtecke nach Ihrer Höhe filtern
    max_height = max(h for x, y, w, h in rectangles)
    filtered_rectangles = [(x, y, w, h) for x, y, w, h in rectangles if h >= max_height * 0.50]

    #for (x, y, w, h) in rectangles:
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (int(color[0]),int(color[1]),int(color[2])), 2)

    # neues Rechteck berechnen
    x_min = min([x for x, y, w, h in filtered_rectangles])
    y_min = min([y for x, y, w, h in filtered_rectangles])
    x_max = max([x + w for x, y, w, h in filtered_rectangles])
    y_max = max([y + h for x, y, w, h in filtered_rectangles])

    area = (x_max - x_min) * (y_max - y_min)
    if area < 15000 or area > 130000: # kleine und große Rechtecke filtern
        return None

    return x_min, y_min, x_max - x_min, y_max - y_min

# Breite Konturen, welche größer als der threshold sind teilen
def split_wide_contours(contours, max_width):
    new_contours = []

    for contour in contours:
        # Berechne die Bounding Box der aktuellen Kontur
        x, y, w, h = cv2.boundingRect(contour)

        if w > max_width:   # ist zu breit
            mid_x = x + w // 2
            # Kontur in 2 Hälften teilen
            left_contour = contour[np.where(contour[:, :, 0] < mid_x)]
            right_contour = contour[np.where(contour[:, :, 0] >= mid_x)]
            if abs(len(left_contour) - len(right_contour) > 0) > 50:
                continue
            if len(left_contour) > 0:
                new_contours.append(left_contour)
            if len(right_contour) > 0:
                new_contours.append(right_contour)
        elif w > max_width * 2:
            mid_x_1 = x + w // 3
            mid_x_2 = x + 2 * w // 3
            left_contour = contour[np.where(contour[:, :, 0] < mid_x_1)]
            middle_contour = contour[mid_x_2 > np.where(contour[:, :, 0] >= mid_x_1)]
            right_contour = contour[np.where(contour[:, :, 0] >= mid_x_2)]
            if len(left_contour) > 0:
                new_contours.append(left_contour)
            if len(middle_contour) > 0:
                new_contours.append(middle_contour)
            if len(right_contour) > 0:
                new_contours.append(right_contour)
        else:
            new_contours.append(contour)

    return new_contours, len(new_contours) > len(contours)


# Konturen filtern indem die Anzahl der weißen Pixel, welche Sie bedecken genutzt wird
def filter_contours_by_white_area(contours, fgmask, min_white_pixels=10000):
    filtered_contours = []

    for contour in contours:
        mask = np.zeros_like(fgmask, dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1) # Kontur in die Maske füllen
        white_pixels = cv2.countNonZero(cv2.bitwise_and(fgmask, mask))
        if white_pixels >= min_white_pixels: # Anzahl der weißen Pixel prüfen
            filtered_contours.append(contour)

    return filtered_contours

# Person detektieren
def detectPerson(frame, subtractor):
    # BGS anwenden
    fgmask = subtractor.apply(frame)

    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5)))
    fgmask = cv2.dilate(fgmask, np.ones((9, 9)))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS, (9, 9)))
    for _ in range(5):
        fgmask = cv2.erode(fgmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))


    # Teil des unteren Bildes ignorieren
    fgmask[-(fgmask.shape[0] // 10):, :] = 0


    # Konturen finden
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours, hasSplit = split_wide_contours(contours, 400)
    contours = filter_contours_by_white_area(contours, fgmask)

    candidates = []
    for cnt in contours:
        box = minimal_covering_rectangles(cnt, frame)
        if box is not None:
            candidates.append(box)

    if len(candidates) == 0:
        return [], fgmask

    reach = 10 if not hasSplit else 0
    merged_candidates = merge_overlapping_boxes(candidates, reach)

    return merged_candidates, fgmask

# Detektion einzeichnen
def showDetection(frame, sub):
    boundingBoxes = detectPerson(frame, sub)
    if boundingBoxes is not None:
        for i, box in enumerate(boundingBoxes):
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow('Detection', frame)
