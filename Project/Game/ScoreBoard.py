import pygame


class ScoreBoard:
    def __init__(self, numTeams):
        super(ScoreBoard, self).__init__()
        self.points = [0] * numTeams
        self.font = pygame.font.Font(None, 36)  # Schriftgröße anpassen nach Bedarf
        self.spacing = 100 // numTeams

    def draw(self, screen):
        width = screen.get_width()
        spacing = width - 100

        for i, score in enumerate(self.points):
            x = 100 + i * spacing
            y = 10
            if i == 0:
                score_text = self.font.render(f"Team Apple : {score} ", True, (255, 0, 0))
            else:
                score_text = self.font.render(f"Team Banana : {score} ", True, (255, 255, 0))
            screen.blit(score_text, (x - i * (score_text.get_width() + 100), y))

    def changeScore(self, player_index, points):
        if 0 <= player_index < len(self.points):
            self.points[player_index] += points

    def resetScores(self):
        # Alle Scores auf 0 zurücksetzen
        self.points = [0] * len(self.points)
