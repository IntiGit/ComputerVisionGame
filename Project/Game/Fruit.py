import pygame
import random

# Klasse für die Früchte
class Fruit:
    def __init__(self, fruit_type, speed, screen):
        self.screen = screen
        self.fruit_type = fruit_type
        self.image = pygame.image.load(f'Assets/{fruit_type}.png')
        self.rect = self.image.get_rect()
        self.rect.x = random.randint(0, screen.get_width() - self.rect.width)
        self.rect.y = 0
        self.speed = speed


    # Position aktualisieren
    def update_pos_Y(self):
        self.rect.y += self.speed

    # Frucht zeichnen
    def draw(self, surface):
        bounding_box = self.rect.copy()
        pygame.draw.rect(surface, (255, 0, 255), bounding_box, 2)
        surface.blit(self.image, self.rect)