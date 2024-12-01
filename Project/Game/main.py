import numpy as np
import cv2
import pygame
import random
from Fruit import Fruit
from ScoreBoard import ScoreBoard
from Player import Player
from Project.Tracking.DetectionHandler import DetectionHandler

############################################################
# In dieser Datei das Spiel starten durch die main Methode #
############################################################

background_image = pygame.image.load("Assets/background1.jpg")
SCREEN_WIDTH = background_image.get_width()
SCREEN_HEIGHT = background_image.get_height()
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]
useCamera = False
MAX_FRUITS = 3
SPAWN_INTERVAL = 1000

numTeams = 2
playersPerTeam = 1

playerSprites = [pygame.image.load("Assets/playerSpriteRed.png"),
                 pygame.image.load("Assets/playerSpriteYellow.png")]

def initCamera(video_path=None, useCam=False):
    cap = cv2.VideoCapture(0 if useCam else video_path)
    if not cap.isOpened():
        raise ValueError("Camera or video file could not be opened.")
    return cap

def getCameraFrame(cap, detection_handler=None):
    ret, frame = cap.read()
    if not ret:
        return None, None
    if detection_handler:
        fg_frame, positions = detection_handler(frame)
        return frame, positions
    return frame, []



def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("Computer Vision Game")
    fps = 30
    clock = pygame.time.Clock()
    #cap = initCamera(screen)
    detection_handler = DetectionHandler()
    cap = initCamera("../Reflection_5.mp4")

    players = []
    for i in range(numTeams * playersPerTeam):
        sprite = playerSprites[i]
        posX = (i + 1) * (screen.get_width() // (numTeams * playersPerTeam + 1)) - (sprite.get_width() // 2)
        players.append(
            Player(
                posX,
                screen.get_height() - sprite.get_height(),
                sprite,
                "apple" if i // playersPerTeam == 0 else "banana"))

    fruits = []
    last_spawn_time = pygame.time.get_ticks()
    scoreBoard = ScoreBoard(numTeams)

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # press 'esc' to quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        screen.blit(background_image, (0, 0))


        frame, positions = getCameraFrame(cap, detection_handler)
        if frame is None:
            break
        if positions:
            for player, pos in zip(players, positions):
                player.rect.x = int(pos[0] / frame.shape[1] * screen.get_width())  # Skaliere Position

        if len(fruits) < MAX_FRUITS and (current_time - last_spawn_time) > SPAWN_INTERVAL:
            fruit_type = random.choice(['banana', 'apple'])
            speed = random.randint(2, 5)

            new_fruit = Fruit(fruit_type, speed, screen)
            fruits.append(new_fruit)

            # Update the last spawn  time
            last_spawn_time = current_time

        # Update and draw each fruit
        for fruit in fruits:
            fruit.update_pos_Y()
            fruit.draw(screen)

        scoreBoard.draw(screen)

        for player in players:
            player.draw(screen)
            scoreChange, fruitIndex = player.checkCollision(fruits)
            if scoreChange != 0:
                playerIndex = 0 if player.team == "apple" else 1
                scoreBoard.changeScore(playerIndex, scoreChange)
                fruits.pop(fruitIndex)

        pygame.display.update()
        clock.tick(fps)

    pygame.quit()
    cap.release()


if __name__ == '__main__':
    main()
