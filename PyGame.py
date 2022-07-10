import pygame


class PyGame:

    def __init__(self):
        pygame.init()
        self.src = pygame.display.set_mode((400, 400))

    def draw(self, x, y, color):
        pygame.draw.circle(self.src, color, (x, y), 1)
        pygame.display.flip()

    def reset_fill(self):
        self.src.fill((0, 0, 0))
        pygame.display.flip()
