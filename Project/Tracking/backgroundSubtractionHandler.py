import cv2

class BackgroundSubtractionHandler:
    def __init__(self):

        # Subtraktoren initialisieren
        self.subtractors = {
            "MOG2": cv2.createBackgroundSubtractorMOG2(history=126, varThreshold=50, detectShadows=False),
            "KNN": cv2.createBackgroundSubtractorKNN(history=107, dist2Threshold=0, detectShadows=False),
            "CNT": cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=10, maxPixelStability=102,
                                                              useHistory=True, isParallel=True),
            "GSOC": cv2.bgsegm.createBackgroundSubtractorGSOC(0, 20, 0.0035, 0.01, 5, 0.01, 0.0022, 0.1, 0.1, 0.0004, 0.005)
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
        return self.current_subtractor.apply(frame)

    def post_processing(self, frame):

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.medianBlur(frame, 5)
        return frame