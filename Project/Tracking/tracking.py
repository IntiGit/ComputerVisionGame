import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class Track:
    def __init__(self, track_id, bbox, initial_histogram, color):
        self.track_id = track_id
        self.bbox = bbox
        self.last_detection = bbox
        self.average_histogram = initial_histogram
        self.histograms = [initial_histogram]  # Liste, um die letzten Histogramme zu speichern
        self.color = color  # Unique color for this track
        self.age = 0  # Track age (frames since created)
        self.movementDir = 0

        self.kf = cv2.KalmanFilter(2, 1)
        self.kf.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1e-2, 0], [0, 1e-2]], dtype=np.float32)  # Prozessrauschen
        self.kf.measurementNoiseCov = np.array([[1e-1]], dtype=np.float32)  # Messrauschen
        self.kf.errorCovPost = np.array([[1, 0], [0, 1]], dtype=np.float32)  # Anfangsunsicherheit

        x, y, w, h = bbox
        self.kf.statePost = np.array([x, 0], dtype=np.float32)

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

    def predict(self):
        # Kalman-Prediction für x
        predicted = self.kf.predict()
        predicted_x = int(predicted[0, 0])
        predicted_v = int(predicted[1, 0])

        # Verwende vorheriges y, w, h und vorhergesagtes x
        x, y, w, h = self.bbox
        self.bbox = (int(predicted_x - self.age * predicted_v * 1.5), y, w, h)


class PersonTracker:
    def __init__(self):
        self.tracks = []
        self.last_detections = []
        self.next_id = 0
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)  # Pre-generate random colors

    def calculate_histogram(self, image, sub, bbox):
        x, y, w, h = bbox
        roi = image[y:y + h, x:x + w]
        sub_roi = sub[y:y + h, x:x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.threshold(sub_roi, 127, 255, cv2.THRESH_BINARY)[1]
        hist = cv2.calcHist([hsv], [0, 1], mask, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        return hist

    def calculate_cost_matrix(self, tracks, detections, image, sub):
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                hist_det = self.calculate_histogram(image, sub, det)
                hist_similarity = cv2.compareHist(track.average_histogram, hist_det, cv2.HISTCMP_BHATTACHARYYA)
                dist = abs(track.bbox[0] - det[0]) / 200
                if dist > 1:
                    dist = 100
                cost_matrix[i, j] = hist_similarity #+ dist

        return cost_matrix

    def update(self, detections : list, image, sub):
        if len(self.tracks) == 0:
            # Create new tracks for all detections
            for det in detections:
                hist = self.calculate_histogram(image, sub, det)
                self.tracks.append(Track(self.next_id, det, hist, self.colors[self.next_id % 100]))
                self.next_id += 1
            return

        for track in self.tracks:
            track.predict()

        # Calculate cost matrix
        cost_matrix_hist = self.calculate_cost_matrix(self.tracks, detections, image, sub)
        # Solve assignment problem
        track_indices, det_indices = linear_sum_assignment(cost_matrix_hist)

        # Update matched tracks
        unmatched_tracks = set(range(len(self.tracks)))
        unmatched_detections = set(range(len(detections)))

        for t_idx, d_idx in zip(track_indices, det_indices):
            if cost_matrix_hist[t_idx, d_idx] > 0.6:
                continue

            d_bbox = detections[d_idx]
            t_bbox = self.tracks[t_idx].bbox
            if abs(t_bbox[0] - d_bbox[0]) > 400:
                continue
            hist = self.calculate_histogram(image, sub, d_bbox)
            self.tracks[t_idx].update(d_bbox, hist)

            unmatched_tracks.discard(t_idx)
            unmatched_detections.discard(d_idx)

        # Handle unmatched tracks (aging or removal)
        for t_idx in unmatched_tracks:
            self.tracks[t_idx].age += 1
            self.tracks[t_idx].predict()

        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.age < 80]

        # Create new tracks for unmatched detections
        for d_idx in unmatched_detections:
            bbox = detections[d_idx]
            skip = False
            for t in self.tracks:
                if abs((bbox[0] + bbox[2] // 2) - (t.bbox[0] + t.bbox[2] // 2)) < 80:
                    skip = True
            if skip:
                continue
            if bbox[2] * bbox[3] > 15000 and bbox[3] / bbox[2] > 2.25:
                hist = self.calculate_histogram(image, sub, bbox)
                self.tracks.append(Track(self.next_id, bbox, hist, self.colors[self.next_id % 100]))
                self.next_id += 1

        return self.tracks

    def draw_tracks(self, image):
        for track in self.tracks:
            x, y, w, h = track.bbox
            color = [int(c) for c in track.color]
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            cv2.putText(image, f'ID: {track.track_id}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)