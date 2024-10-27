import numpy as np
import cv2
import time

###################################################
# In dieser Datei die Methoden zum Tracken testen #
###################################################

cap = cv2.VideoCapture("Assets/traffic.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = 1 / fps
subtractor = cv2.createBackgroundSubtractorKNN()

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = subtractor.apply(frame)
    _, frame = cv2.threshold(frame, 200, 255, cv2.THRESH_BINARY)
    frame = cv2.erode(frame, None, iterations=1)
    frame = cv2.dilate(frame, None, iterations=1)

    cv2.imshow("Videoansicht", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

