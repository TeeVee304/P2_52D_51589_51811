import cv2 as cv
import numpy as np

MAX_FRAMES = 300
"""
def background(video_path):

    cap = cv.VideoCapture(video_path)
    count = 0
    avg = None

    while count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        f = frame.astype(np.float32)
        if avg is None:
            avg = f
        else:
            avg += f

        count += 1

    cap.release()

    avg /= count
    return avg.astype(np.uint8)
"""

class BackgroundExtractor:
    """Extrai background de um vídeo.

    Métodos:
    - build_median_background: amostra frames e usa mediana (mais robusto)
    - build_mog2_background: usa MOG2 para casos com movimento persistente
    """
    def __init__(self, max_samples: int = 300, sample_step: int = 5):
        self.max_samples = max_samples
        self.sample_step = sample_step

    def build_median_background(self, video_path: str) -> np.ndarray:
        cap = cv.VideoCapture(video_path)
        total = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frames = []
        idx = 0
        collected = 0
        while collected < self.max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % self.sample_step == 0:
                frames.append(frame.astype(np.uint8))
                collected += 1
            idx += 1
        cap.release()

        if len(frames) == 0:
            raise RuntimeError("Nenhum frame lido para construir background")

        # Mediana ao longo do eixo 0
        med = np.median(np.stack(frames, axis=0), axis=0).astype(np.uint8)
        return med

    def build_mog2_background(self, video_path: str) -> np.ndarray:
        cap = cv.VideoCapture(video_path)
        fgbg = cv.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            fgbg.apply(frame)
            count += 1
            if count > 1000:
                break
        cap.release()
        # Aproximação do background (usando método getBackgroundImage quando disponível)
        bg = fgbg.getBackgroundImage()
        if bg is None:
            raise RuntimeError("MOG2 não retornou imagem de background")
        return bg