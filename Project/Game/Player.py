import pygame
import numpy as np


class Player(pygame.sprite.Sprite):
    # -----------------------------------------------------------
    # init class
    def __init__(self, posX, posY, sprite, team):
        super(Player, self).__init__()
        self.sprite = pygame.transform.flip(sprite, True, False)
        self.rect = self.sprite.get_rect()
        self.rect.x = posX
        self.rect.y = posY
        self.speed = 10
        self.flip = False

    def draw(self, screen):
        image_to_draw = pygame.transform.flip(self.sprite, self.flip, False)
        screen.blit(image_to_draw, self.rect)
        bounding_box = self.rect.copy()  # Kopie des Rechtecks erstellen
        pygame.draw.rect(screen, (255, 0, 0), bounding_box, 2)

    def update(self, keys, screen):
        prevDir = self.flip
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
            self.flip = False
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
            self.flip = True
        self.rect.x = max(0, min(screen.get_width() - self.rect.width, self.rect.x))
        self.draw(screen)
