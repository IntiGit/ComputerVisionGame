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
        self.team = team

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

    def checkCollision(self, fruits):
        print(fruits)
        scoreChange = 0
        toRemove = set()
        for i, fruit in enumerate(fruits):
            if self.rect.colliderect(fruit.rect):
                if fruit.fruit_type == self.team:
                    scoreChange += 1
                else:
                    scoreChange -= 1
                toRemove.add(i)

        return scoreChange, toRemove
