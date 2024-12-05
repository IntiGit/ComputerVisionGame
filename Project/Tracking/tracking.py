import cv2
import numpy as np


class PersonTracker:
    def __init__(self):
        # Kalman-Filter mit nur horizontaler Komponente
        self.kalman = cv2.KalmanFilter(2, 1)  # Zustand: [x, dx], Messung: [x]
        self.kalman.measurementMatrix = np.array([[1, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 1],
                                                 [0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0],
                                                [0, 1]], np.float32) * 0.03
        self.last_detection = None
        self.last_y = 0

    def update(self, detection):
        if detection is not None:
            x, y, w, h = detection
            center_x = x + w / 2
            self.last_y = y + h // 2  # y bleibt konstant aus der letzten Messung
            measurement = np.array([[np.float32(center_x)]])
            self.kalman.correct(measurement)
            self.last_detection = detection
        else:
            self.last_detection = None

        # Vorhersage machen
        prediction = self.kalman.predict()
        pred_x = prediction[0]

        # Bounding Box nur horizontal verschieben, y bleibt konstant
        return int(pred_x - 50), int(self.last_y - 100), 100, 200

    def draw_prediction(self, frame, bbox):
        x, y, w, h = bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
