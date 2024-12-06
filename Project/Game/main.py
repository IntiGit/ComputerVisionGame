import numpy as np
import cv2
import pygame
import random
from Fruit import Fruit
from ScoreBoard import ScoreBoard
from Player import Player
from Project.Tracking.backgroundSubtractionHandler import BackgroundSubtractionHandler

############################################################
# In dieser Datei das Spiel starten durch die main Methode #
############################################################

background_image = pygame.image.load("Assets/background1.jpg")
SCREEN_WIDTH = background_image.get_width()
SCREEN_HEIGHT = background_image.get_height()
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]
useCamera = False
MAX_FRUITS = 2
SPAWN_INTERVAL = 1000

playerSprites = [pygame.image.load("Assets/playerSpriteRed.png"),
                 pygame.image.load("Assets/playerSpriteYellow.png")]


def initCamera(screen, useCam=False):
    cap = cv2.VideoCapture("")
    if useCam:
        cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())
    return cap


def getCameraFrame(cap):
    ret, cameraFrame = cap.read()
    imgRGB = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2RGB)
    imgRGB = np.rot90(imgRGB)
    gameFrame = pygame.surfarray.make_surface(imgRGB).convert()
    return gameFrame


def main():
    pygame.init()
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("Computer Vision Game")
    fps = 30
    clock = pygame.time.Clock()
    cap = initCamera(screen)

    sprite = playerSprites[0]
    posX = screen.get_width() // 2 - sprite.get_width() // 2
    player = Player(posX, screen.get_height() - sprite.get_height(), sprite, "apple")

    fruits = []
    last_spawn_time = pygame.time.get_ticks()

    scoreBoard = ScoreBoard(1)

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

        if len(fruits) < MAX_FRUITS and (current_time - last_spawn_time) > SPAWN_INTERVAL:
            fruit_type = random.choice(['banana', 'apple'])
            speed = random.randint(2, 5)

            new_fruit = Fruit(fruit_type, speed, screen)
            fruits.append(new_fruit)

            # Update the last spawn  time
            last_spawn_time = current_time

        # Update and draw each fruit
        toRemove = set()
        for fruit in fruits:
            fruit.update_pos_Y()
            if fruit.rect.y > screen.get_height():
                toRemove.add(fruit)
                continue
            fruit.draw(screen)
        for f in toRemove:
            fruits.remove(f)

        scoreBoard.draw(screen)

        if useCamera:
            screen.blit(getCameraFrame(cap), (0, 0))

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


if __name__ == '__main__':
    main()
