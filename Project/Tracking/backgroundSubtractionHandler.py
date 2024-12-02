import cv2
import numpy as np

class BackgroundSubtractionHandler:
    def __init__(self):

        # Subtraktoren initialisieren
        self.subtractors = {
            "MOG2": cv2.createBackgroundSubtractorMOG2(),
            "KNN": cv2.createBackgroundSubtractorKNN(),
            "CNT": cv2.bgsegm.createBackgroundSubtractorCNT(),
            "GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC(0, 20, 0.0035, 0.01,
                                                              5, 0.01, 0.0022,
                                                              0.1, 0.1,
                                                              0.0004,
                                                              0.005)
        }

        self.current_subtractor_name = "MOG2"
        self.current_subtractor = self.subtractors[self.current_subtractor_name]

    def set_subtractor(self, name):
        if name in self.subtractors:
            self.current_subtractor_name = name
            self.current_subtractor = self.subtractors[name]
        else:
            raise ValueError(f"Unbekannter Subtraktor: {name}")

    def apply_subtractor(self, frame):

         if self.current_subtractor_name == "MOG2":
            self.current_subtractor.setHistory(500)
            self.current_subtractor.setVarThreshold(50)
            self.current_subtractor.setShadowThreshold(0.2)
            self.current_subtractor.setShadowValue(255)
            self.current_subtractor.setNMixtures(5)
            self.current_subtractor.setBackgroundRatio(0.8)
            self.current_subtractor.setDetectShadows(True)

         return self.current_subtractor.apply(frame)

    def post_processing(self, frame):
        # Morphologische Operationen
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (14, 14))
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        #frame = cv2.medianBlur(frame, 2)

        return frame
