import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
import collections


# Klasse, für den einzelnen Tracks einer Person
class Track:
    def __init__(self, track_id, bbox, initial_histogram, color):
        self.track_id = track_id
        self.bbox = bbox
        self.last_detection = bbox
        self.average_histogram = initial_histogram
        self.histograms = [initial_histogram]  # Liste, um die letzten Histogramme zu speichern
        self.color = color  # Einzigartige Farbe für diesen Track
        self.age = 0  # Alter des Tracks (Anzahl Frames seit Erstellung)
        self.movementDir = 0

        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1e-2, 0], [0, 1e-2]], dtype=np.float32)  # Prozessrauschen
        self.kf.measurementNoiseCov = np.array([[1e-1]], dtype=np.float32)  # Messrauschen
        self.kf.errorCovPost = np.array([[1, 0], [0, 1]], dtype=np.float32)  # Anfangsunsicherheit

        x, y, w, h = bbox
        self.kf.statePost = np.array([x, 0], dtype=np.float32)

        self.overlap_counter = 0
        self.last_reliable_hist = initial_histogram.copy()
        self.last_reliable_bbox = bbox
        self.overlap_threshold = 0.3
        self.velocity_history = collections.deque(maxlen=10)
        self.direction = None  # Bewegungsrichtung
        self.stable_count = 0  # Zähler für stabile Frames
        self.confidence = 1.0  # Konfidenz des Tracks

    # Aktualisiert den Track mit neuer Bounding Box und neuem Histogramm
    def update(self, bbox, new_histogram):
        old_x = self.bbox[0]
        dx = bbox[0] - old_x

        # Aktualisiere Bewegungsrichtung
        if abs(dx) > 5:
            current_direction = 'right' if dx > 0 else 'left'
            if self.direction == current_direction:
                self.stable_count += 1
            else:
                self.stable_count = 0
            self.direction = current_direction

        # Speichere Geschwindigkeit
        self.velocity_history.append(dx)

        # Prüfe Bewegungskonsistenz
        if len(self.velocity_history) > 5:
            velocity_std = np.std(list(self.velocity_history))
            if velocity_std > 30:  # Hohe Varianz in der Bewegung
                self.confidence *= 0.9
            else:
                self.confidence = min(1.0, self.confidence + 0.1)

        # Rest des Updates
        self.bbox = bbox
        self.kf.correct(np.array([[np.float32(bbox[0])]]))
        self.histograms.append(new_histogram)
        if len(self.histograms) > 10:
            self.histograms.pop(0)
        self.average_histogram = sum(self.histograms) / len(self.histograms)
        self.age = 0
        return True

    # nächste Position der BBox nach Kalman-Filter
    def predict(self):
        # Kalman-Prediction für x
        predicted = self.kf.predict()
        predicted_x = int(predicted[0, 0])  # Vorhergesagte x-Position
        predicted_v = int(predicted[1, 0])  # Vorhergesagte Geschwindigkeit

        # Verwende vorheriges y, w, h und vorhergesagtes x
        x, y, w, h = self.bbox
        self.bbox = (int(predicted_x - self.age * predicted_v * 1.5), y, w, h)


class PersonTracker:
    def __init__(self):
        self.tracks = []  # alle Tracks
        self.last_detections = []
        self.next_id = 0  # # Nächste verfügbare ID für neue Tracks
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)  # Pre-generate random colors

    def calculate_overlap_ratio(self, bbox1, bbox2):
        """Berechnet das Überlappungsverhältnis zweier Bounding Boxes"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2

        # Berechne Überlappungsbereich
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = w1 * h1
        area2 = w2 * h2
        return intersection / min(area1, area2)

    def calculate_histogram(self, image, sub, bbox, overlapped=False):
        x, y, w, h = bbox
        roi = image[y:y + h, x:x + w]
        sub_roi = sub[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Bei Überlappung: Fokus auf charakteristische Merkmale
        if overlapped:
            # Oberer Bereich (Kopf und Schultern) bekommt mehr Gewicht
            h_reduced = int(h * 0.3)  # Reduziert auf oberen 30%
            hsv = hsv[:h_reduced, :]
            sub_roi = sub_roi[:h_reduced, :]

        # Vertikale Aufteilung in mehr Regionen für bessere Unterscheidung
        h_curr = hsv.shape[0]
        w_curr = hsv.shape[1]

        # Aufteilung in 6 Regionen (3 vertikal, 2 horizontal)
        h_third = h_curr // 3
        w_half = w_curr // 2

        regions = [
            # Oberer Bereich (Kopf)
            (hsv[0:h_third, 0:w_half], sub_roi[0:h_third, 0:w_half]),
            (hsv[0:h_third, w_half:w_curr], sub_roi[0:h_third, w_half:w_curr]),
            # Mittlerer Bereich (Oberkörper)
            (hsv[h_third:2 * h_third, 0:w_half], sub_roi[h_third:2 * h_third, 0:w_half]),
            (hsv[h_third:2 * h_third, w_half:w_curr], sub_roi[h_third:2 * h_third, w_half:w_curr]),
            # Unterer Bereich (Beine)
            (hsv[2 * h_third:h_curr, 0:w_half], sub_roi[2 * h_third:h_curr, 0:w_half]),
            (hsv[2 * h_third:h_curr, w_half:w_curr], sub_roi[2 * h_third:h_curr, w_half:w_curr])
        ]

        # Angepasste Gewichtung
        if overlapped:
            weights = [0.3, 0.3, 0.2, 0.2, 0.0, 0.0]  # Fokus auf obere Bereiche
        else:
            # Dynamische Gewichtung basierend auf der Maskenfläche
            weights = []
            total_area = w_curr * h_curr
            for region, mask in regions:
                area = np.count_nonzero(mask) / total_area
                weights.append(max(0.05, area))  # Mindestgewicht von 5%
            # Normalisierung der Gewichte
            weights = np.array(weights) / np.sum(weights)
            # Erhöhe Gewichtung für obere Bereiche
            weights[:2] *= 1.5
            weights = weights / np.sum(weights)

        region_hists = []
        for (region, mask), weight in zip(regions, weights):
            if region.size == 0:
                continue

            # Verwende H und S Kanäle
            h_channel = region[:, :, 0]
            s_channel = region[:, :, 1]
            v_channel = region[:, :, 2]

            # Ignoriere Pixel mit sehr niedrigem S oder V Wert
            valid_mask = mask.copy()
            valid_mask[s_channel < 30] = 0  # Ignoriere niedrige Sättigung
            valid_mask[v_channel < 30] = 0  # Ignoriere dunkle Bereiche

            # Berechne separate Histogramme für H und S
            h_hist = cv2.calcHist([h_channel], [0], valid_mask, [30], [0, 180])
            s_hist = cv2.calcHist([s_channel], [0], valid_mask, [32], [0, 256])

            # Normalisiere beide Histogramme
            h_hist = cv2.normalize(h_hist, h_hist, 0, 1, cv2.NORM_MINMAX).flatten()
            s_hist = cv2.normalize(s_hist, s_hist, 0, 1, cv2.NORM_MINMAX).flatten()

            # Gewichte und kombiniere die Histogramme
            # H-Kanal bekommt mehr Gewicht (0.7) als S-Kanal (0.3)
            combined_hist = np.concatenate([h_hist * 0.7, s_hist * 0.3])
            combined_hist *= weight
            region_hists.append(combined_hist)

        final_hist = np.sum(region_hists, axis=0)
        return final_hist.astype(np.float32)

    def calculate_cost_matrix(self, tracks, detections, image, sub):
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # Berechne Histogramm-Ähnlichkeit
                hist_det = self.calculate_histogram(image, sub, det)
                hist_similarity = cv2.compareHist(track.average_histogram, hist_det, cv2.HISTCMP_CORREL)

                # Räumliche Distanz
                dx = abs(track.bbox[0] - det[0])
                dy = abs(track.bbox[1] - det[1])
                dist_penalty = np.exp(-(dx * dx + dy * dy) / (2 * 200 * 200))

                # Größenvergleich
                size_ratio = (det[2] * det[3]) / (track.bbox[2] * track.bbox[3])
                size_penalty = np.exp(-abs(1 - size_ratio))

                # Kombinierte Kosten
                cost = (0.7 * (1 - hist_similarity) +  # Hauptgewicht auf Farbe
                        0.2 * (1 - dist_penalty) +  # Geringeres Gewicht auf Distanz
                        0.1 * (1 - size_penalty))  # Kleinstes Gewicht auf Größe

                cost_matrix[i, j] = cost

        return cost_matrix

    def update(self, detections: list, image, sub):
        if len(self.tracks) == 0:
            # Erstelle neue Tracks für alle Detektionen
            for det in detections:
                hist = self.calculate_histogram(image, sub, det)
                self.tracks.append(Track(self.next_id, det, hist, self.colors[self.next_id % 100]))
                self.next_id += 1
            return self.tracks

        # Vorhersage für alle Tracks
        for track in self.tracks:
            track.predict()

        # Berechne Zuordnungskosten
        cost_matrix = self.calculate_cost_matrix(self.tracks, detections, image, sub)

        # Löse Zuordnungsproblem
        track_indices, det_indices = linear_sum_assignment(cost_matrix)

        # Update zugeordnete Tracks
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))

        # Finde überlappende Detektionen
        overlapping_dets = set()
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i + 1:], i + 1):
                if self.calculate_overlap_ratio(det1, det2) > 0.3:
                    overlapping_dets.add(i)
                    overlapping_dets.add(j)

        # Update Tracks
        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix[t_idx, d_idx] > 0.7:
                continue

            d_bbox = detections[d_idx]
            t_bbox = self.tracks[t_idx].bbox

            # Prüfe auf Überlappung
            is_overlapped = d_idx in overlapping_dets

            if is_overlapped:
                self.tracks[t_idx].overlap_counter += 1
            else:
                self.tracks[t_idx].overlap_counter = 0

            # Berechne Histogramm unter Berücksichtigung der Überlappung
            hist = self.calculate_histogram(image, sub, d_bbox, is_overlapped)

            # Bei Überlappung: Konservativeres Update
            if is_overlapped:
                # Mische neues und altes Histogramm
                alpha = 0.3  # Geringerer Einfluss des neuen Histogramms
                hist = cv2.addWeighted(self.tracks[t_idx].last_reliable_hist, 1 - alpha, hist, alpha, 0)
            else:
                # Speichere zuverlässiges Histogramm
                self.tracks[t_idx].last_reliable_hist = hist.copy()

            # Update Track
            self.tracks[t_idx].update(d_bbox, hist)
            unmatched_tracks.discard(t_idx)
            unmatched_detections.discard(d_idx)

        # Behandle nicht zugeordnete Tracks
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].age += 1
            self.tracks[t_idx].predict()

        # Entferne alte Tracks
        self.tracks = [t for t in self.tracks if t.age < 60]  # Kürzere Lebensdauer

        # Erstelle neue Tracks für nicht zugeordnete Detektionen
        for d_idx in unmatched_detections:
            bbox = detections[d_idx]

            # Prüfe auf nahe existierende Tracks
            if any(abs(t.bbox[0] - bbox[0]) < 100 for t in self.tracks):
                continue

            # Prüfe Personenproportionen
            if (bbox[2] * bbox[3] > 15000 and
                    2.0 < bbox[3] / bbox[2] < 3.5):  # Typisches Seitenverhältnis für Personen

                hist = self.calculate_histogram(image, sub, bbox)
                self.tracks.append(Track(self.next_id, bbox, hist, self.colors[self.next_id % 100]))
                self.next_id += 1

        return self.tracks

    # Zeichne die Tracks auf das Bild
    def draw_tracks(self, image):
        for track in self.tracks:
            x, y, w, h = track.bbox
            color = [int(c) for c in track.color]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, f'ID: {track.track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)