import numpy as np
import cv2
import pygame

from Player import Player

############################################################
# In dieser Datei das Spiel starten durch die main Methode #
############################################################

SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
SCREEN = [SCREEN_WIDTH, SCREEN_HEIGHT]


def main():

    # init pygame
    pygame.init()

    # set display size and caption
    screen = pygame.display.set_mode(SCREEN)
    pygame.display.set_caption("Computer Vision Game")

    # init game clock
    fps = 30
    clock = pygame.time.Clock()

    # opencv - init webcam capture
    cap = cv2.VideoCapture(0)
    # set width & height to screen size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen.get_width())
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen.get_height())

    # init player
    player = Player(screen.get_width() / 2, screen.get_height() / 2)

    # example variable for game score
    gameScore = 0

    # -------------
    # -- main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # press 'esc' to quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        # -- opencv & viz image
        ret, cameraFrame = cap.read()
        imgRGB = cv2.cvtColor(cameraFrame, cv2.COLOR_BGR2RGB)
        # image needs to be rotated for pygame
        imgRGB = np.rot90(imgRGB)
        # convert image to pygame and visualize
        gameFrame = pygame.surfarray.make_surface(imgRGB).convert()
        screen.blit(gameFrame, (0, 0))

        # -- update & draw object on screen
        player.update(pygame.key.get_pressed())
        screen.blit(player.surf, player.rect)

        # -- add Text on screen (e.g. score)
        textFont = pygame.font.SysFont("arial", 26)
        textExample = textFont.render(f'Score: {gameScore}', True, (255, 0, 0))
        screen.blit(textExample, (20, 20))

        # update entire screen
        pygame.display.update()
        # set clock
        clock.tick(fps)

    # quit game
    pygame.quit()

    # release capture
    cap.release()


if __name__ == '__main__':
    main()