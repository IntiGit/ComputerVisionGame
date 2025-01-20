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
video = "Turn_Around"
videoPath = "C:/Users/Timo/Desktop/CV Videos/edited/MOT/" + video + ".mp4"

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

    sprite1 = playerSprites[0]
    player1 = Player(sprite1.get_width(), screen.get_height() - sprite1.get_height(), sprite1, "apple", 0)

    sprite2 = playerSprites[1]
    player2 = Player(screen.get_width() - sprite2.get_width(), screen.get_height() - sprite2.get_height(), sprite2, "banana", 1)

    fruits = []
    last_spawn_time = pygame.time.get_ticks()

    # Punkteanzeige initialisieren
    scoreBoard = ScoreBoard()

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
    frameCount = 0
    running = cap.isOpened()
    while running:
        """Tracking"""
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        detections, bgs = detect.detectPerson(frame, sub)  # Bounding-Boxen von Personen detektieren
        tracks = tracker.update(detections, frame, bgs)  # Tracker aktualisieren
        tracker.draw_tracks(frame)  # Tracks visualisieren

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

        frameCount += 1

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
        if tracks is not None:
            matchedA = matchedB = False
            for t in tracks:
                if t.track_id == player1.track_id:
                    player1.update(t.bbox[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH), screen)
                    matchedA = True
                if t.track_id == player2.track_id:
                    player2.update(t.bbox[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH), screen)
                    matchedB = True

            if not matchedA:
                for t in tracks:
                    if t.track_id != player2.track_id:
                        player1.update(t.bbox[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH), screen)
                        player1.track_id = t.track_id

            if not matchedB:
                for t in tracks:
                    if t.track_id != player1.track_id:
                        player2.update(t.bbox[0] / cap.get(cv2.CAP_PROP_FRAME_WIDTH), screen)
                        player2.track_id = t.track_id

        # Kollisionen überprüfen und Punkte anpassen
        scoreChangeA, toRemoveA = player1.checkCollision(fruits)
        scoreChangeB, toRemoveB = player2.checkCollision(fruits)

        if len(toRemoveA) != 0:
            scoreBoard.changeScore(0, scoreChangeA)
            for fruitIndex in toRemoveA:
                fruits.pop(fruitIndex)

        if len(toRemoveB) != 0:
            scoreBoard.changeScore(0, scoreChangeB)
            for fruitIndex in toRemoveB:
                fruits.pop(fruitIndex)

        pygame.display.update()
        clock.tick(fps)

    # print(np.mean(metric.results))

    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
