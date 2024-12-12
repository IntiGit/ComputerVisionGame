import csv
import math
import cv2

class Metric:
    def __init__(self, csvFile):
        self.csvFile = csvFile
        self.data = self.loadCSVFile()
        self.results = []

    #CSV Datei (GroundTruth) laden
    def loadCSVFile(self):
        data = []
        try:
            with open(self.csvFile, mode='r', newline='') as file:
                reader = csv.reader(file)
                # Optionally, skip header if CSV has one
                next(reader)  # Uncomment this line if the first row is a header
                for row in reader:
                    data.append(row)
        except FileNotFoundError:
            print(f"Error: Die Datei {self.csvFile} wurde nicht gefunden.")
        except Exception as e:
            print(f"Ein Fehler ist aufgetreten: {e}")

        return data

    def get_row_by_frame(self, frameNr):
        # Zeile uschen die der frameNr entspricht
        for row in self.data:
            if int(row[0]) == frameNr:
                # Wenn frameNr übereinstimmt, Werte zurückgeben
                frame_data = (
                    int(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4])
                )
                return frame_data
        return None

    # RSE berechnen
    def relative_size_error(self, frameCount, pred):
        gt = self.get_row_by_frame(frameCount)
        if gt is None:
            return

        _, _, _, w_gt, h_gt = gt
        _, _, w_pred, h_pred = pred

        area_gt = w_gt * h_gt
        area_pred = w_pred * h_pred

        if area_gt == 0 or area_pred == 0:
            return

        rse = (area_pred - area_gt) / area_gt

        self.results.append(rse)

    # IoU berechnen
    def IoU(self, frameCount, pred):
        gt = self.get_row_by_frame(frameCount)
        if gt is None:
            return

        fn, x1_gt, y1_gt, w_gt, h_gt = gt
        x1_pred, y1_pred, w_pred, h_pred = pred

        # Berechne rechte untere Ecken
        x2_gt = x1_gt + w_gt
        y2_gt = y1_gt + h_gt
        x2_pred = x1_pred + w_pred
        y2_pred = y1_pred + h_pred

        # Berechne die Grenzen des Intersections-Rechtecks
        x1_inter = max(x1_gt, x1_pred)
        y1_inter = max(y1_gt, y1_pred)
        x2_inter = min(x2_gt, x2_pred)
        y2_inter = min(y2_gt, y2_pred)

        # Berechne die Fläche der Intersection (falls vorhanden)
        inter_width = max(0, x2_inter - x1_inter)
        inter_height = max(0, y2_inter - y1_inter)
        intersection_area = inter_width * inter_height

        # Berechne die Fläche der Union
        area_gt = w_gt * h_gt
        area_pred = w_pred * h_pred
        if area_gt == 0:
            return

        union_area = area_gt + area_pred - intersection_area

        # Verhindere Division durch Null
        if union_area == 0:
            return

        # Berechne IoU
        iou = intersection_area / union_area

        self.results.append(iou)

    # DE berechnen
    def displacement_error(self, frameCount, pred):
        gt = self.get_row_by_frame(frameCount)
        if gt is None: # keine Ground-Truth-Daten
            return

        _, x_gt, y_gt, w_gt, h_gt = gt
        x_pred, y_pred, _, _ = pred
        if w_gt * h_gt == 0:
            return

        de = math.sqrt((x_pred - x_gt) ** 2 + (y_pred - y_gt) ** 2) # euklidischen Abstand zwischen den Mittelpunkten

        self.results.append(de)

