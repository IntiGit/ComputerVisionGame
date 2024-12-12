from collections import deque

import cv2
import numpy as np


class PersonTracker:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(2, 1)
        self.kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 1],
                                                 [0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.03
        self.last_detection = None
        self.last_y = 0

        self.sizes = deque(maxlen=10)

    def update(self, detection):
        if detection is not None:
            x, y, w, h = detection
            if self.last_detection is not None:
                if abs(x - self.last_detection[0]) > 250:
                    detection = self.last_detection
                    x, y, w, h = detection
            center_x = x + w / 2
            self.last_y = y + h // 2
            measurement = np.array([[np.float32(center_x)]])
            self.kalman.correct(measurement)
            self.last_detection = detection

            # Speichern der Größe
            self.sizes.append((w, h))
        else:
            self.last_detection = None

        # Vorhersage machen
        prediction = self.kalman.predict()
        pred_x = prediction[0]

        # Berechnung der gewichteten Durchschnittsgröße
        if self.sizes:
            weights = np.exp(np.linspace(0, -1, len(self.sizes)))
            weights /= weights.sum()
            avg_w = int(np.dot(weights, [size[0] for size in self.sizes]))
            avg_h = int(np.dot(weights, [size[1] for size in self.sizes]))
        else:
            avg_w, avg_h = 0, 0  # Standardwerte

        # Bounding Box erstellen
        return int(pred_x - avg_w / 2), int(self.last_y - avg_h / 2), avg_w, avg_h

    # Zeichnen der Vorhersage
    def draw_prediction(self, frame, bbox):
        x, y, w, h = bbox
        if x < 0 or y < 0:
            return
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

