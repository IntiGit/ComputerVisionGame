import numpy as np
import cv2
import time


###################################################
# In dieser Datei die Methoden zum Tracken testen #
###################################################

def postProcessing(frame):
    newframe = cv2.erode(frame, None, iterations=1)
    newframe = cv2.dilate(newframe, None, iterations=1)
    return newframe


def applySubtractor(frame, subtractor):
    return subtractor.apply(frame)


# Subtraktoren definieren
subtractors = [
    ("MOG2", cv2.createBackgroundSubtractorMOG2(history=126, varThreshold=50, detectShadows=False)),
    ("KNN", cv2.createBackgroundSubtractorKNN(history=107, dist2Threshold=0, detectShadows=False)),
    ("CNT", cv2.bgsegm.createBackgroundSubtractorCNT(minPixelStability=10, maxPixelStability=102,
                                                     useHistory=True, isParallel=True)),
    ("GSOC", cv2.bgsegm.createBackgroundSubtractorGSOC(0, 20, 0.0035, 0.01, 5,
                                                       0.01, 0.0022, 0.1, 0.1,
                                                       0.0004, 0.0008))
]
subtractorIndex = 0  # Auswahl des Subtraktors
sub = subtractors[subtractorIndex]

# Trackbar-Update-Funktionen für KNN
def update_DetectShadows_KNN(val):
    sub.setDetectShadows(val == 1)

def update_Dist2Threshold_KNN(val):
    sub.setDist2Threshold(val/10)

def update_kNNSamples_KNN(val):
    sub.setkNNSamples(val)

def update_NSamples_KNN(val):
    sub.setNSamples(val)

def update_ShadowThreshold_KNN(val):
    sub.setShadowThreshold(val/10)

def update_ShadowValue_KNN(val):
    sub.setShadowValue(val)

# Trackbar-Update-Funktionen für MOG2
def update_history(val):
    sub.setHistory(val)
def update_varThreshold(val):
    sub.setVarThreshold(val)
def update_shadowThreshold(val):
    sub.setShadowThreshold(val)

# VarMin, VarMax werden dynamisch vom Modell berechnet
def update_VarThresholdGen(val):
   sub.setVarThresholdGen(val)
def update_VarInit(val):
   sub.setVarInit(val)
def update_complexityReductionThreshold(val):
    sub.setComplexityReductionThreshold(val / 10.0)  # Teilt durch 10 für eine feinere Anpassung
def update_nMixtures(val):
    sub.setNMixtures(val)
def update_BackgorundRatio(val):
    sub.setBackgroundRatio(val/10.0) # Teilt durch 10 für eine feinere Anpassung
def update_detectShadows(val):
    sub.setDetectShadows(val == 1)


# Trackbar-Update-Funktionen für CNT
def update_minPixelStability(value):
    sub.setMinPixelStability(value)
def update_maxPixelStability(value):
    sub.setMaxPixelStability(value)
def update_useHistory(value):
    sub.setUseHistory(value)
def update_isParallel(value):
    sub.setIsParallel(value)

# GUI für Trackbars zur dynamischen Anpassung für MOG2
cv2.namedWindow("BS")
''' MOG2
cv2.createTrackbar("History", "BS", 126, 500, update_history)
cv2.createTrackbar("VarThreshold", "BS", 30, 255, update_varThreshold)
cv2.createTrackbar("Detect Shadows", "BS", 0, 1, update_detectShadows)
cv2.createTrackbar("Shadow Threshold", "BS", 94, 100, update_shadowThreshold)
cv2.createTrackbar("Complexity Reduction Threshold", "BS", 9, 100, update_complexityReductionThreshold)
cv2.createTrackbar("N Mixtures", "BS", 8, 10, update_nMixtures)
cv2.createTrackbar("BackgroundRatio", "BS", 7, 100, update_BackgorundRatio)
cv2.createTrackbar("VarInit", "BS", 94, 100, update_VarInit)
cv2.createTrackbar("VarThresholdGen", "BS", 10, 10, update_VarThresholdGen)
'''

''' KNN
cv2.createTrackbar("History", "BS", 107, 200, update_history)
cv2.createTrackbar("Dist2Threshold", "BS", 0, 255, update_Dist2Threshold_KNN)
cv2.createTrackbar("Detect Shadows", "BS", 1, 1, update_DetectShadows_KNN)
cv2.createTrackbar("KNN Samples", "BS", 3, 100, update_kNNSamples_KNN)
cv2.createTrackbar("NSamples", "BS", 25, 100, update_NSamples_KNN)
cv2.createTrackbar("Shadow Value", "BS", 0, 100, update_ShadowValue_KNN)
cv2.createTrackbar("Shadow Threshold", "BS", 9, 100, update_ShadowThreshold_KNN)
'''

''' CNT
cv2.namedWindow("BS")
cv2.createTrackbar("Min Pixel Stability", "BS", 10, 255, update_minPixelStability)
cv2.createTrackbar("Max Pixel Stability", "BS", 102, 500, update_maxPixelStability)
cv2.createTrackbar("Use History", "BS", 1, 1, update_useHistory)
cv2.createTrackbar("Is Parallel", "BS", 1, 1, update_isParallel)
'''

# Wendet den Subtractor (und die Nachverarbeitung) auf das Video an und speichert das Ergebnis
def applyAndSave(input_video_path, output_video_path, applyPostProcessing=False):
    sub = subtractors[subtractorIndex][1]

    match subtractorIndex:
        case 0:
            sub.setShadowThreshold(94)
            sub.setComplexityReductionThreshold(0.05)
            sub.setNMixtures(8)
            sub.setBackgroundRatio(0.7)
            sub.setVarInit(94)
        case 1:
            sub.setkNNSamples(3)
            sub.setNSamples(25)
            sub.setShadowValue(0)
            sub.setShadowThreshold(9)
    # Video einlesen
    cap = cv2.VideoCapture(input_video_path)

    if not cap.isOpened():
        print("Fehler: Konnte das Video nicht laden.")
        return
    # Videoeigenschaften
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 30
    # VideoWriter erstellen
    fourcc = 0x7634706d
    out = cv2.VideoWriter(output_video_path, int(fourcc), float(fps), (frame_width, frame_height), False)
    # Frame für Frame durch das Video gehen
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Hintergrundsubtraktion anwenden
        fg_mask = sub.apply(frame)
        # Nachverarbeitung anwenden
        if applyPostProcessing:
            fg_mask = postProcessing(fg_mask)
        # Das Bild ins Ausgabevideo schreiben
        out.write(fg_mask)

    cap.release()
    out.release()
    print("Das Ergebnisvideo wurde gespeichert.")


def main(output_video_path="adjustments_output.mp4"):
    cap = cv2.VideoCapture("Assets/Videos/Reflection_1.mp4")

    # Videoeigenschaften
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # VideoWriter für Resultate
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height), False)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # FPS
        start_time = time.time()

        # Hintergrundsubtraktion anwenden
        fg_mask = applySubtractor(frame, sub)

        # FPS berechnen
        end_time = time.time()
        fps = 1 / (end_time - start_time) if end_time - start_time > 0 else 0

        # FPS auf dem Frame anzeigen
        cv2.putText(fg_mask, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("BS", fg_mask)

        # Frame in Ausgabevideo schreiben
        out.write(fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()  # Speicherung abschließen
    cv2.destroyAllWindows()
    print(f"Das Video wurde als '{output_video_path}' gespeichert")


# Starte den Hauptprozess und speichere das Ergebnisvideo
# main("./Video.mp4")

# Subtraktor anwenden und Resultat speichern
#MOG2
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/MOG2/raw/Ref1_MOG2_raw.mp4")
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/MOG2/raw/Ref5_MOG2_raw.mp4")
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/MOG2/raw/RW1_MOG2_raw.mp4")
#
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/MOG2/post_processing/Ref1_MOG2_pp.mp4", True)
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/MOG2/post_processing/Ref5_MOG2_pp.mp4", True)
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/MOG2/post_processing/RW1_MOG2_pp.mp4", True)
# subtractorIndex += 1
#KNN
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/KNN/raw/Ref1_KNN_raw.mp4")
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/KNN/raw/Ref5_KNN_raw.mp4")
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/KNN/raw/RW1_KNN_raw.mp4")
#
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/KNN/post_processing/Ref1_KNN_pp.mp4", True)
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/KNN/post_processing/Ref5_KNN_pp.mp4", True)
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/KNN/post_processing/RW1_KNN_pp.mp4", True)
# subtractorIndex += 1
#CNT
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/CNT/raw/Ref1_CNT_raw.mp4")
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/CNT/raw/Ref5_CNT_raw.mp4")
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/CNT/raw/RW1_CNT_raw.mp4")
#
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/CNT/post_processing/Ref1_CNT_pp.mp4", True)
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/CNT/post_processing/Ref5_CNT_pp.mp4", True)
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/CNT/post_processing/RW1_CNT_pp.mp4", True)
# subtractorIndex += 1
#GSOC
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/GSOC/raw/Ref1_GSOC_raw.mp4")
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/GSOC/raw/Ref5_GSOC_raw.mp4")
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/GSOC/raw/RW1_GSOC_raw.mp4")
#
# applyAndSave("Assets/Videos/Reflection_1.mp4", "Assets/Results/GSOC/post_processing/Ref1_GSOC_pp.mp4", True)
# applyAndSave("Assets/Videos/Reflection_5.mp4", "Assets/Results/GSOC/post_processing/Ref5_GSOC_pp.mp4", True)
# applyAndSave("Assets/Videos/RoomWindow_1.mp4", "Assets/Results/GSOC/post_processing/RW1_GSOC_pp.mp4", True)
