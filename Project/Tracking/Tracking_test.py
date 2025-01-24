import collections
import cv2
import numpy as np
import detection as detect
import tracking as track
import json


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
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Tracking", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Testet und speichert das Ergebnis des Trackings
def testMOT(video_path, json_path):
    cap = cv2.VideoCapture(video_path)

    sub = cv2.createBackgroundSubtractorMOG2()
    sub.setBackgroundRatio(0.8)
    sub.setDetectShadows(True)
    sub.setShadowThreshold(0.2)
    sub.setShadowValue(255)
    sub.setHistory(800)
    sub.setNMixtures(7)
    sub.setVarThreshold(50)

    tracker = track.PersonTracker()

    with open(json_path, 'r') as f:
        groundtruth = json.load(f)

    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter("./GT_PRED/" + video + "_Result.mp4",
                          fourcc,
                          fps,
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                           int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    frame_id = 0
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255), ]
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        detections, bgs = detect.detectPerson(frame, sub)  # Bounding-Boxen von Personen detektieren
        tracker.update(detections, frame, bgs)  # Tracker aktualisieren
        tracker.draw_tracks(frame, colors)  # Tracks visualisieren

        frame_data = next((item for item in groundtruth if item["frame_id"] == frame_id), None)
        if frame_data:
            # Bounding Boxes und IDs auf das Frame zeichnen
            for obj in frame_data["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                object_id = obj["id"]
                class_id = obj["class_id"]
                confidence = obj["confidence"]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                text = f"ID: {object_id}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("Tracking", frame)

        out.write(frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


# Berechnet IoU
def calc_iou(trackBox, truthBox):
    x1_t, y1_t, w1, h1 = trackBox
    x1_truth, y1_truth, w2, h2 = truthBox

    # Berechne die Koordinaten der unteren rechten Ecken
    x2_t = x1_t + w1
    y2_t = y1_t + h1
    x2_truth = x1_truth + w2
    y2_truth = y1_truth + h2

    # Berechne die Koordinaten der überlappenden Box
    x1_overlap = max(x1_t, x1_truth)
    y1_overlap = max(y1_t, y1_truth)
    x2_overlap = min(x2_t, x2_truth)
    y2_overlap = min(y2_t, y2_truth)

    # Berechne die Fläche der Überlappung
    overlap_width = max(0, x2_overlap - x1_overlap)
    overlap_height = max(0, y2_overlap - y1_overlap)
    overlap_area = overlap_width * overlap_height

    # Berechne die Flächen der beiden Boxen
    area1 = w1 * h1
    area2 = w2 * h2

    # Berechne die Union-Fläche
    union_area = area1 + area2 - overlap_area

    # Vermeide Division durch Null und berechne den IoU
    if union_area == 0:
        return 0.0
    return overlap_area / union_area


def metricAnalysis(video_path, json_path):
    cap = cv2.VideoCapture(video_path)

    sub = cv2.createBackgroundSubtractorMOG2()
    sub.setBackgroundRatio(0.8)
    sub.setDetectShadows(True)
    sub.setShadowThreshold(0.2)
    sub.setShadowValue(255)
    sub.setHistory(800)
    sub.setNMixtures(7)
    sub.setVarThreshold(50)

    tracker = track.PersonTracker()

    with open(json_path, 'r') as f:
        groundtruth = json.load(f)

    frame_id = 0

    ious = {}
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        detections, bgs = detect.detectPerson(frame, sub)  # Bounding-Boxen von Personen detektieren
        tracks = tracker.update(detections, frame, bgs)  # Tracker aktualisieren

        frame_data = next((item for item in groundtruth if item["frame_id"] == frame_id), None)
        if frame_data:
            for obj in frame_data["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                object_id = obj["id"]
                class_id = obj["class_id"]
                confidence = obj["confidence"]

                if tracks is not None:
                    for t in tracks:
                        if t.track_id == object_id:
                            ious[object_id] = calc_iou(t.bbox, (x1, y1, x2 - x1, y2 - y1))

        cv2.imshow("Frame", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    print(sum(ious.values()) / len(ious))


video = "Many_2"
video_path = "C:/Users/Timo/Desktop/CV Videos/edited/MOT/" + video + ".mp4"
json_path = "./Truths/groundtruth_" + video + ".json"
metricAnalysis(video_path, json_path)
#testMOT(video_path, json_path)
