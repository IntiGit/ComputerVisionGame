import numpy as np
import cv2
import time


###################################################
# In dieser Datei die Methoden zum Tracken testen #
###################################################


def postProccesing(frame):
    _, newframe = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
    newframe = cv2.erode(newframe, None, iterations=1)
    newframe = cv2.dilate(newframe, None, iterations=1)
    return newframe


def applySubtractor(frame, subtractor):
    return subtractor.apply(frame)


subtractors = [("MOG2", cv2.createBackgroundSubtractorMOG2()),
               ("KNN", cv2.createBackgroundSubtractorKNN()),
               ("MOG", cv2.bgsegm.createBackgroundSubtractorMOG()),
               ("CNT", cv2.bgsegm.createBackgroundSubtractorCNT()),
               ("GMG", cv2.bgsegm.createBackgroundSubtractorGMG()),
               ("GSOC", cv2.bgsegm.createBackgroundSubtractorGSOC()),
               ("LSBP", cv2.bgsegm.createBackgroundSubtractorLSBP())]
subtractorIndex = 0


def main():
    cap = cv2.VideoCapture("Assets/Videos/traffic.mp4")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame = applySubtractor(frame, subtractors[subtractorIndex][1])
        #frame = postProccesing(frame)
        cv2.imshow(subtractors[subtractorIndex][0], frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()
