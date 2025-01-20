import pygame


# Klasse für das ScoreBoard
class ScoreBoard:
    def __init__(self):
        super(ScoreBoard, self).__init__()
        self.points = [0, 0]
        self.font = pygame.font.Font(None, 36)  # Schriftgröße anpassen nach Bedarf
        self.spacing = 50

    # Punkte und Team zeichnen
    def draw(self, screen):
        width = screen.get_width()
        spacing = width - 100

        score_textA = self.font.render(f"Team Apple : {self.points[0]} ", True, (255, 0, 0))
        score_textB = self.font.render(f"Team Banana : {self.points[1]} ", True, (255, 255, 0))
        xA = 100
        xB = screen.get_width() - 100 - score_textB.get_width()
        y = 10
        screen.blit(score_textA, (xA, y))
        screen.blit(score_textB, (xB, y))

    # Punktestand aktualisieren
    def changeScore(self, player_index, points):
        if 0 <= player_index < len(self.points):
            self.points[player_index] += points

    def resetScores(self):
        # Alle Scores auf 0 zurücksetzen
        self.points = [0] * len(self.points)
