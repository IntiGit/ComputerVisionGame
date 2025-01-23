import cv2
import json
from ultralytics import YOLO

# YOLO-Modell laden
model = YOLO('yolov8n.pt')

# Video-Dateipfade
root_path = "C:/Users/Timo/Desktop/CV Videos/edited/MOT/"
video = "_"
video_path = root_path + video + ".mp4"

# Video einlesen
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Fehler: Video konnte nicht geöffnet werden.")
    exit()

# Video-Eigenschaften
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Video: {video_path}")
print(f"Auflösung: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")

# Groundtruth-Daten speichern
groundtruth = []

# Frame-Iterator
frame_id = 0
curID = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO-Inferenz auf dem aktuellen Frame
    results = model.predict(frame, conf=0.5, verbose=False)

    # Daten für das aktuelle Frame sammeln
    frame_data = {
        "frame_id": frame_id,
        "objects": []
    }
    nextID = 0
    # Bounding Boxen und IDs speichern
    for result in results[0].boxes.data.tolist():
        # Bounding Box und Klasseninformationen extrahieren
        x1, y1, x2, y2, confidence, class_id = result
        if class_id != 0:
            continue
        object_data = {
            "id": len(frame_data["objects"]),  # Lokale ID innerhalb des Frames
            "class_id": int(class_id),
            "confidence": float(confidence),
            "bbox": [int(x1), int(y1), int(x2), int(y2)]
        }
        frame_data["objects"].append(object_data)
        nextID += 1

        # Bounding Box auf Frame zeichnen (optional für Visualisierung)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {object_data['id']}, XY: {object_data['bbox'][0]} {object_data['bbox'][1]} \nFrame{frame_id}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Groundtruth-Liste um Frame-Daten erweitern
    if len(frame_data["objects"]) != 0:
        groundtruth.append(frame_data)

    # Frame anzeigen (optional)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Frame-Zähler erhöhen
    frame_id += 1

# Video und Fenster schließen
cap.release()
cv2.destroyAllWindows()

# Groundtruth-Daten als JSON speichern
output_path = f"./Truths/groundtruth_{video}.json"
with open(output_path, "w") as f:
    json.dump(groundtruth, f, indent=4)

print(f"Groundtruth wurde erfolgreich gespeichert: {output_path}")
