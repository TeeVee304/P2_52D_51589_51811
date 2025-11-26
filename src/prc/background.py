import cv2 as cv

class Background:
    """Classe responsável pela estimativa e manutenção do modelo de fundo numa sequência de vídeo."""

    def estimate(self, frame):
        """"""