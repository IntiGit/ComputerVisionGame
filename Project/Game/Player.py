import pygame
import numpy as np

# Klasse für den / die Spieler
class Player(pygame.sprite.Sprite):
    def __init__(self, posX, posY, sprite, team):
        super(Player, self).__init__()
        self.sprite = pygame.transform.flip(sprite, True, False)
        self.rect = self.sprite.get_rect()
        self.rect.x = posX
        self.rect.y = posY
        self.speed = 10
        self.flip = False
        self.team = team

    # Spieler zeichnen
    def draw(self, screen):
        image_to_draw = pygame.transform.flip(self.sprite, self.flip, False)
        screen.blit(image_to_draw, self.rect)
        bounding_box = self.rect.copy()  # Kopie des Rechtecks erstellen
        pygame.draw.rect(screen, (255, 0, 0), bounding_box, 2)

    # Akutelle X-Position des Spielers aktulisieren
    def update(self, bboxX, screen):
        self.rect.x = bboxX * screen.get_width()
        self.rect.x = max(0, min(screen.get_width() - self.rect.width, self.rect.x))
        self.draw(screen)

    # Auf Kollision mit Frucht prüfen
    def checkCollision(self, fruits):
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
