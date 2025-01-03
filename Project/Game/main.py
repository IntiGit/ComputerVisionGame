import collections

import numpy as np
import cv2
import pygame
import random
import Project.Tracking.tracking as track
import Project.Tracking.detection as detect
import Project.Tracking.metrics as mt
from Fruit import Fruit
from ScoreBoard import ScoreBoard
from Player import Player

############################################################
# In dieser Datei das Spiel starten durch die main Methode #
############################################################

# Hintergrundbild und Spielbildschirm-Größe festlegen
background_image = pygame.image.load("Assets/background1.jpg")
SCREEN_WIDTH = background_image.get_width()
SCREEN_HEIGHT = background_image.get_height()
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]
MAX_FRUITS = 2
SPAWN_INTERVAL = 1000

# Spieler-Sprites
playerSprites = [pygame.image.load("Assets/playerSpriteRed.png"),
                 pygame.image.load("Assets/playerSpriteYellow.png")]

# Pfad für Video
video = "Brick_2"
videoPath = "_" + video + ".mp4"

# Zum Messen von DE, RSE, IoU
# metric = mt.Metric("../Tracking/Truths/groundTruth_" + video + ".csv")

# Video speichern
def writeToOutput(out, frame, gt_box, pred_box):
    _, x_gt, y_gt, w_gt, h_gt = gt_box
    x_p, y_p, w_p, h_p = pred_box
    cv2.rectangle(frame, (x_gt, y_gt), (x_gt + w_gt, y_gt + h_gt), (117, 0, 255), 2)
    cv2.rectangle(frame, (x_p, y_p), (x_p + w_p, y_p + h_p), (0, 255, 0), 2)
    out.write(frame)

# Erstellt neue Früchte
def spawnFruit(fruits, current_time, last_spawn_time, screen):
    if len(fruits) < MAX_FRUITS and (current_time - last_spawn_time) > SPAWN_INTERVAL:
        fruit_type = random.choice(['banana', 'apple'])
        speed = random.randint(7, 10)

        new_fruit = Fruit(fruit_type, speed, screen)
        fruits.append(new_fruit)

        # Update the last spawn  time
        return current_time
    return last_spawn_time

# Aktualisieren und Entfernen von Früchten
def updateFruits(fruits, screen):
    toRemove = set()
    for fruit in fruits:
        fruit.update_pos_Y()
        if fruit.rect.y > screen.get_height():
            toRemove.add(fruit)
            continue
        fruit.draw(screen)
    for f in toRemove:
        fruits.remove(f)


# Subtractor für unsere BGS mit MOG2 einrichten
def setupSubtractor():
    sub = cv2.createBackgroundSubtractorMOG2()
    sub.setBackgroundRatio(0.8)
    sub.setDetectShadows(True)
    sub.setShadowThreshold(0.2)
    sub.setShadowValue(255)
    sub.setHistory(500)
    sub.setNMixtures(5)
    sub.setVarThreshold(50)
    return sub


def main():
    """Game Setup"""
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("Computer Vision Game")
    fps = 30
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(videoPath)
    if videoPath is None:  # Wenn kein Video angegeben, dann Kamera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())

    sprite = playerSprites[0]
    posX = screen.get_width() // 2 - sprite.get_width() // 2
    player = Player(posX, screen.get_height() - sprite.get_height(), sprite, "apple")

    fruits = []
    last_spawn_time = pygame.time.get_ticks()

    # Punkteanzeige initialisieren
    scoreBoard = ScoreBoard(1)

    """Videos speichern"""
    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec für .mp4
    # out = cv2.VideoWriter("_" + video + "_Result.mp4",
    #                       fourcc,
    #                       fps,
    #                       (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    #                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    """Tracking Setup"""
    sub = setupSubtractor()
    tracker = track.PersonTracker()

    y_buffer = collections.deque(maxlen=50) # vergangende y-Werte speichern
    h_buffer = collections.deque(maxlen=50) # vergangende h-Werte speichern
    avgY, avgH = 0, 0

    detectionCount = 0
    frameCount = 0
    new_values = []
    descriptor = None

    last_detection = None
    running = cap.isOpened()
    while running:
        """Tracking"""
        ret, frame = cap.read()
        if not ret:
            break
        frameCount += 1
        # Detektion einer Person im aktuellen Frame
        detection = detect.detectPerson(frame, sub, descriptor, last_detection)

        # Buffer aktualisieren
        if detection:
            x, y, w, h = detection
            last_detection = detection
            # Wenn die Box groß genug ist, wird wahrscheinlich die ganze Person drin sein
            if w * h > 70000:
                _, descriptor = detect.extract_orb_features(frame, (x, y, w, h))

            if detectionCount % 100 == 0:
                new_values = []

            if len(new_values) < 30:
                new_values.append((y, h))

            if len(new_values) == 30:
                y_buffer.extend(val[0] for val in new_values)
                h_buffer.extend(val[1] for val in new_values)
                avgY = np.median(y_buffer)  # Berechne Median-Werte aus Buffer
                avgH = np.median(h_buffer)
                new_values.append(0)

            detectionCount += 1

            # Korrigieren von y/h um Außreißer zu unterdrücken
            if avgY is not None and avgH is not None and len(y_buffer) > 30:
                if abs(y - avgY) > 80 or abs(h - avgH) > 160:
                    detection = (x, avgY, w, avgH)

        bbox = tracker.update(detection)

        # writeToOutput(out, frame, metric.get_row_by_frame(frameCount), bbox)
        # metric.relative_size_error(frameCount, bbox)

        tracker.draw_prediction(frame, bbox)
        cv2.imshow("Cam", frame)

        """Game"""
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # press 'esc' to quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
        # hintergrundbild zeichnen
        screen.blit(background_image, (0, 0))

        last_spawn_time = spawnFruit(fruits, current_time, last_spawn_time, screen)

        updateFruits(fruits, screen)

        scoreBoard.draw(screen)

        # Spieler-Position basierend auf Tracker aktualisieren
        player.update(bbox[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH), screen)
        # Kollisionen überprüfen und Punkte anpassen
        scoreChange, toRemove = player.checkCollision(fruits)

        if len(toRemove) != 0:
            scoreBoard.changeScore(0, scoreChange)
            for fruitIndex in toRemove:
                fruits.pop(fruitIndex)

        pygame.display.update()
        clock.tick(fps)

    # print(np.mean(metric.results))

    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
