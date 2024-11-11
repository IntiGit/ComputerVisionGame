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


def applyAndSave(input_video_path, output_video_path, subtractor):
    # Video einlesen
    cap = cv2.VideoCapture(input_video_path)

    # Überprüfen, ob das Video erfolgreich geladen wurde
    if not cap.isOpened():
        print("Fehler: Konnte das Video nicht laden.")
        return

    # Videoeigenschaften (Größe, FPS) holen
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter initialisieren, um das Ergebnisvideo zu speichern
    fourcc = 0x7634706d
    out = cv2.VideoWriter(output_video_path, int(fourcc), float(fps), (frame_width, frame_height), False)

    # Frame für Frame durch das Video iterieren
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Hintergrundsubtraktion anwenden
        fg_mask = subtractor.apply(frame)

        # Das maskierte Bild ins Ausgabevideo schreiben
        out.write(fg_mask)

    # Ressourcen freigeben
    cap.release()
    out.release()
    print("Das Ergebnisvideo wurde erfolgreich gespeichert.")



subtractors = [("MOG2", cv2.createBackgroundSubtractorMOG2(history=170, varThreshold=95, detectShadows=False)),
               ("KNN", cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=True)),
               ("MOG", cv2.bgsegm.createBackgroundSubtractorMOG(history=170, nmixtures=5, backgroundRatio=0.5, noiseSigma=1.5)),
               ("CNT", cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, maxPixelStability=60, isParallel=True)),
               ("GMG", cv2.bgsegm.createBackgroundSubtractorGMG(initializationFrames=50, decisionThreshold=0.7)),
               ("GSOC", cv2.bgsegm.createBackgroundSubtractorGSOC(mc=10, nSamples=10, replaceRate=0.01)),
               ("LSBP", cv2.bgsegm.createBackgroundSubtractorLSBP(mc=10, Tlower=2, Tupper=32))]
subtractorIndex = 6




def main():
    cap = cv2.VideoCapture("film.mov")
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # FPS calculation
        start_time = time.time()

        # Apply background subtraction
        fg_mask = applySubtractor(frame, subtractors[subtractorIndex][1])

        # frame = postProccesing(frame)

        # Calculate FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time) if end_time - start_time > 0 else 0

        # Display FPS on frame
        cv2.putText(fg_mask, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(subtractors[subtractorIndex][0], fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


#main()
input_video_path = 'Assets/Videos/traffic.mp4'
output_video_path = 'Assets/Results/_X_.mp4'
applyAndSave(input_video_path, output_video_path,  subtractors[subtractorIndex][1])
