import collections

import numpy as np
import cv2
import pygame
import random
import Project.Tracking.tracking as track
import Project.Tracking.detection as detect
from Fruit import Fruit
from ScoreBoard import ScoreBoard
from Player import Player

############################################################
# In dieser Datei das Spiel starten durch die main Methode #
############################################################

background_image = pygame.image.load("Assets/background1.jpg")
SCREEN_WIDTH = background_image.get_width()
SCREEN_HEIGHT = background_image.get_height()
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]
MAX_FRUITS = 2
SPAWN_INTERVAL = 1000

playerSprites = [pygame.image.load("Assets/playerSpriteRed.png"),
                 pygame.image.load("Assets/playerSpriteYellow.png")]

videoPath = "C:/Users/Timo/Desktop/CV Videos/edited/SOT/Brick_1.mp4"


def spawnFruit(fruits, current_time, last_spawn_time, screen):
    if len(fruits) < MAX_FRUITS and (current_time - last_spawn_time) > SPAWN_INTERVAL:
        fruit_type = random.choice(['banana', 'apple'])
        speed = random.randint(2, 5)

        new_fruit = Fruit(fruit_type, speed, screen)
        fruits.append(new_fruit)

        # Update the last spawn  time
        return current_time
    return last_spawn_time


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
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("Computer Vision Game")
    fps = 30
    clock = pygame.time.Clock()
    cap = cv2.VideoCapture(videoPath)
    if videoPath is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())

    sprite = playerSprites[0]
    posX = screen.get_width() // 2 - sprite.get_width() // 2
    player = Player(posX, screen.get_height() - sprite.get_height(), sprite, "apple")

    fruits = []
    last_spawn_time = pygame.time.get_ticks()

    scoreBoard = ScoreBoard(1)

    '''Tracking Setup'''
    sub = setupSubtractor()
    tracker = track.PersonTracker()

    y_buffer = collections.deque(maxlen=50)
    h_buffer = collections.deque(maxlen=50)
    avgY, avgH = 0, 0

    frameCount = 0
    new_values = []
    descriptor = None

    running = cap.isOpened()
    while running:
        '''Tracking'''
        ret, frame = cap.read()
        if not ret:
            break

        detection = detect.detectPerson(frame, sub, descriptor)

        if detection:
            x, y, w, h = detection

            if w * h > 70000:
                _, descriptor = detect.extract_orb_features(frame, (x, y, w, h))

            if frameCount % 100 == 0:
                new_values = []

            if len(new_values) < 30:
                new_values.append((y, h))

            if len(new_values) == 30:
                y_buffer.extend(val[0] for val in new_values)
                h_buffer.extend(val[1] for val in new_values)
                avgY = np.median(y_buffer)
                avgH = np.median(h_buffer)
                new_values.append(0)

            frameCount += 1

            if avgY is not None and avgH is not None and len(y_buffer) > 30:
                if abs(y - avgY) > 80 or abs(h - avgH) > 160:
                    detection = (x, avgY, w, avgH)

        bbox = tracker.update(detection)
        tracker.draw_prediction(frame, bbox)
        cv2.imshow("Cam", frame)

        '''Game'''
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # press 'esc' to quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.blit(background_image, (0, 0))

        last_spawn_time = spawnFruit(fruits, current_time, last_spawn_time, screen)

        updateFruits(fruits, screen)

        scoreBoard.draw(screen)

        player.update(pygame.key.get_pressed(), screen)

        scoreChange, toRemove = player.checkCollision(fruits)

        if len(toRemove) != 0:
            scoreBoard.changeScore(0, scoreChange)
            for fruitIndex in toRemove:
                fruits.pop(fruitIndex)

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
