from ultralytics import YOLO
import cv2
import csv

# Lade das Modell
model = YOLO('yolov8n.pt')
root_path = "C:/Users/Timo/Desktop/CV Videos/edited/SOT/"
video = "Moving_Occlusion_1"
video_path = root_path + video + ".mp4"

# Öffne das Video
cap = cv2.VideoCapture(video_path)

# Öffne die CSV-Datei im Schreibmodus
with open('./Truths/groundTruth_' + video + '.csv', 'w', newline='') as csvfile:

    fieldnames = ['FrameNr', 'x', 'y', 'w', 'h']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()  # Schreibe die Headerzeile

    frame_number = 0  # Frame-Nummer

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1  # Erhöhe die Frame-Nummer

        results = model.track(frame, persist=True)

        detections = results[0].boxes
        detected = False  # Flag, um zu überprüfen, ob eine Person erkannt wurde

        for det in detections:
            if det.cls == 0:  # Nur Personen (Klasse 0)
                x1, y1, x2, y2 = det.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                # Berechne Breite und Höhe der Bounding Box
                w = x2 - x1
                h = y2 - y1

                # Zeichne die Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                # Speichern der Bounding Box in der CSV-Datei
                writer.writerow({'FrameNr': frame_number, 'x': x1, 'y': y1, 'w': w, 'h': h})
                detected = True
                break

        if not detected:
            # Wenn keine Bounding Box gefunden wurde, speichere die Information
            writer.writerow({'FrameNr': frame_number, 'x': 0, 'y': 0, 'w': 0, 'h': 0})

        # Zeige das Frame mit den Bounding Boxen
        cv2.imshow("Frame", frame)

        # Beende bei Drücken von 'q'
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()