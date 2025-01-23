import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

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

    # Aktualisiert den Track mit neuer Bounding Box und neuem Histogramm
    def update(self, bbox, new_histogram):
        self.bbox = bbox
        self.last_detection = bbox
        self.kf.correct(np.array([[np.float32(bbox[0])]]))

        # Füge das neue Histogramm hinzu und halte die Liste auf maximal 10 Einträge
        self.histograms.append(new_histogram)
        if len(self.histograms) > 10:
            self.histograms.pop(0)

        # Berechne das Durchschnittshistogramm
        self.average_histogram = sum(self.histograms) / len(self.histograms)
        self.age = 0

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
        self.tracks = [] # alle Tracks
        self.last_detections = []
        self.next_id = 0 # # Nächste verfügbare ID für neue Tracks
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)  # Pre-generate random colors

    # Berechnet das Farb-Histogramm eines Bereichs
    def calculate_histogram(self, image, sub, bbox):
        x, y, w, h = bbox
        roi = image[y:y + h, x:x + w]  # Region of Interest
        sub_roi = sub[y:y + h, x:x + w]  # Subtrahierte Maske
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV) # Konvertiere zu HSV-Farbraum
        mask = cv2.threshold(sub_roi, 127, 255, cv2.THRESH_BINARY)[1] # Binäre Maske
        hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256]) # Berechne Histogramm
        cv2.normalize(hist, hist)  # Normalisiere das Histogramm
        return hist

    # Berechnet die Kostenmatrix für Zuordnungen zwischen Tracks und Erkennungen
    def calculate_cost_matrix(self, tracks, detections, image, sub):
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                hist_det = self.calculate_histogram(image, sub, det)
                hist_similarity = cv2.compareHist(track.average_histogram, hist_det, cv2.HISTCMP_BHATTACHARYYA) # Vergleich der Histogramme
                dist = abs(track.bbox[0] - det[0]) / 200 # Abstand in X-Richtung
                if dist > 1:
                    dist = 100  # Strafe für große Distanzen
                cost_matrix[i, j] = hist_similarity  # Setze Kostenwert

        return cost_matrix

    # Aktualisiert die Tracks
    def update(self, detections : list, image, sub):
        if len(self.tracks) == 0:
            # Create new tracks for all detections
            for det in detections:
                hist = self.calculate_histogram(image, sub, det)
                self.tracks.append(Track(self.next_id, det, hist, self.colors[self.next_id % 100]))
                self.next_id += 1
            return

        for track in self.tracks:
            track.predict()  # sagt die nächste Position des Tracks vorher

        # Berechne Kostenmatrix
        cost_matrix_hist = self.calculate_cost_matrix(self.tracks, detections, image, sub)
        # Löse das Zuordnungsproblem
        track_indices, det_indices = linear_sum_assignment(cost_matrix_hist)

        # Update matched tracks
        unmatched_tracks = set(range(len(self.tracks)))  # Nicht zugeordnete Tracks
        unmatched_detections = set(range(len(detections))) # Nicht zugeordnete Erkennungen

        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix_hist[t_idx, d_idx] > 0.6:  # Verwerfe unähnliche Zuordnungen
                continue

            d_bbox = detections[d_idx]
            t_bbox = self.tracks[t_idx].bbox
            if abs(t_bbox[0] - d_bbox[0]) > 400:  # Verwerfe weit entfernte Zuordnungen
                continue
            hist = self.calculate_histogram(image, sub, d_bbox)
            self.tracks[t_idx].update(d_bbox, hist)

            unmatched_tracks.discard(t_idx)
            unmatched_detections.discard(d_idx)

        # Behandle nicht zugeordnete Tracks (z. B. altern oder entfernen)
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].age += 1
            self.tracks[t_idx].predict()

        # alte Tracks entfernen
        self.tracks = [t for t in self.tracks if t.age < 80]

        # Erstelle neue Tracks für nicht zugeordnete Erkennungen
        for d_idx in unmatched_detections:
            bbox = detections[d_idx]
            skip = False
            for t in self.tracks:
                if abs((bbox[0] + bbox[2] // 2) - (t.bbox[0] + t.bbox[2] // 2)) < 80:
                    skip = True
            if skip:
                continue
            if bbox[2] * bbox[3] > 15000 and bbox[3] / bbox[2] > 2.25: # Größenverhältnis prüfen
                hist = self.calculate_histogram(image, sub, bbox)
                self.tracks.append(Track(self.next_id, bbox, hist, self.colors[self.next_id % 100]))
                self.next_id += 1

        return self.tracks

    # Zeichne die Tracks auf das Bild
    def draw_tracks(self, image):
        for track in self.tracks:
            x, y, w, h = track.bbox
            color = [int(c) for c in track.color]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f'ID: {track.track_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)