import collections
import cv2
import numpy as np
import Project.Tracking.detection as detect
import Project.Tracking.tracking as track


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
            break

        boundingBox = detect.detectPerson(frame, sub)
        if boundingBox:
            x, y, w, h = boundingBox
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def testTracking(frame,sub, descriptor, frameCount,y_buffer, h_buffer, tracker,avgY, avgH, new_values):

    detection = detect.detectPerson(frame, sub, descriptor)
    if detection:
        x, y, w, h = detection

        if w * h > 70000:
            _, descriptor = detect.extract_orb_features(frame, (x, y, w, h))

        if frameCount % 100 == 0:
            new_values = []

        if len(new_values) < 30:
            new_values.append((y, h))

        if len(new_values) == 30:
            y_buffer.extend(val[0] for val in new_values)
            h_buffer.extend(val[1] for val in new_values)
            avgY = np.median(y_buffer)
            avgH = np.median(h_buffer)
            new_values.append(0)

        frameCount += 1

        if avgY is not None and avgH is not None and len(y_buffer) > 30:
            if abs(y - avgY) > 80 or abs(h - avgH) > 160:
                detection = (x, avgY, w, avgH)

    bbox = tracker.update(detection)
    return bbox




