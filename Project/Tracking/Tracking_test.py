import cv2
import numpy as np
import detection as detect


def testDetection(video_path):
    cap = cv2.VideoCapture(video_path)

    sub = cv2.createBackgroundSubtractorMOG2()
    sub.setBackgroundRatio(0.8)
    sub.setDetectShadows(True)
    sub.setShadowThreshold(0.2)
    sub.setShadowValue(255)
    sub.setHistory(500)
    sub.setNMixtures(5)
    sub.setVarThreshold(50)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video konnte nicht geladen werden")
            break

        boundingBox = detect.detectPerson(frame, sub)
        if boundingBox:
            x, y, w, h = boundingBox
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


testDetection("INSERT_PATH_HERE")
