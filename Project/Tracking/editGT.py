import cv2
import json


def display_groundtruth(video_path, json_path):
    # Video einlesen
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Fehler: Video konnte nicht geöffnet werden.")
        return

    # JSON-Daten einlesen
    with open(json_path, 'r') as f:
        groundtruth = json.load(f)

    frame_id = 0  # Initialisiere Frame-Id

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Daten für das aktuelle Frame abrufen
        frame_data = next((item for item in groundtruth if item["frame_id"] == frame_id), None)

        if frame_data:
            # Bounding Boxes und IDs auf das Frame zeichnen
            for obj in frame_data["objects"]:
                x1, y1, x2, y2 = obj["bbox"]
                object_id = obj["id"]
                class_id = obj["class_id"]
                confidence = obj["confidence"]

                # Bounding Box und Text mit ID, X, Y und Frame-Nummer zeichnen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 2)
                text = f"ID: {object_id}, X: {x1}, Y: {y1}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)


        # Frame anzeigen (optional)
        cv2.putText(frame, f"{frame_id}", (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 6)
        cv2.imshow("Frame with Groundtruth", frame)

        # Mit 'q' das Video stoppen
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2.waitKey(30)

        frame_id += 1

    # Video und Fenster schließen
    cap.release()
    cv2.destroyAllWindows()


# Beispielaufruf der Funktion
video = "Random_1"
video_path = "C:/Users/Timo/Desktop/CV Videos/edited/MOT/" + video + ".mp4"
json_path = "./Truths/groundtruth_" + video + ".json"
display_groundtruth(video_path, json_path)