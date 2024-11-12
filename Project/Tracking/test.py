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


# Subtraktoren definieren
subtractors = [
    ("MOG2", cv2.createBackgroundSubtractorMOG2(history=170, varThreshold=95, detectShadows=False)),
    ("KNN", cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)),
    ("MOG", cv2.bgsegm.createBackgroundSubtractorMOG(history=170, nmixtures=5, backgroundRatio=0.5, noiseSigma=1.5)),
    ("CNT", cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=15, maxPixelStability=60, isParallel=True)),
    ("GSOC", cv2.bgsegm.createBackgroundSubtractorGSOC(mc=10, nSamples=10, replaceRate=0.01))
]
subtractorIndex = 2  # Auswahl des Subtraktors
sub = subtractors[subtractorIndex][1]


# Trackbar-Update-Funktionen
def update_history(val):
    sub.setHistory(val)


def update_varThreshold(val):
    sub.setVarThreshold(val)


def update_shadowThreshold(val):
    sub.setShadowThreshold(val)




#def update_complexityReductionThreshold(val):
 #   sub.setComplexityReductionThreshold(val / 10.0)  # Teilt durch 10 für eine feinere Anpassung (MOG2)


def update_nMixtures(val):
    sub.setNMixtures(val)


def update_detectShadows(val):
    sub.setDetectShadows(val == 1)


# GUI für Trackbars zur dynamischen Anpassung
cv2.namedWindow("BS Adjustments")
cv2.createTrackbar("History", "BS Adjustments", 126, 500, update_history)
cv2.createTrackbar("VarThreshold", "BS Adjustments", 30, 255, update_varThreshold)
cv2.createTrackbar("Detect Shadows", "BS Adjustments", 0, 1, update_detectShadows)
cv2.createTrackbar("Shadow Threshold", "BS Adjustments", 0, 100, update_shadowThreshold)
cv2.createTrackbar("Complexity Reduction", "BS Adjustments", 30, 100, update_complexityReductionThreshold)
cv2.createTrackbar("N Mixtures", "BS Adjustments", 8, 10, update_nMixtures)


def main(output_video_path="adjustments_output.mp4"):
    cap = cv2.VideoCapture("RoomWindow_3.mp4")

    # Videoeigenschaften holen
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter für Live-Resultate initialisieren
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    #out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), False)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # FPS calculation
        start_time = time.time()

        # Hintergrundsubtraktion anwenden
        fg_mask = applySubtractor(frame, sub)

        # FPS berechnen
        end_time = time.time()
        fps = 1 / (end_time - start_time) if end_time - start_time > 0 else 0

        # FPS auf dem Frame anzeigen
        cv2.putText(fg_mask, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("BS Adjustments", fg_mask)

        # Frame in Ausgabevideo schreiben
       # out.write(fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
   # out.release()  # Speicherung abschließen
    cv2.destroyAllWindows()
    print(f"Das Video der Anpassungen wurde als '{output_video_path}' gespeichert.")


# Starte den Hauptprozess und speichere das Ergebnisvideo
main("../Tracking/Results/MOG2_Reflection1.mp4")
