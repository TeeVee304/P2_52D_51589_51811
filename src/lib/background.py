import cv2 as cv
import numpy as np

MAX_FRAMES = 300

def background(video_path):
    """
    Calcula o fundo (background) de um vídeo com base na média de todos os frames de um vídeo.

    Args:
        video_path (str): Diretoria do ficheiro de vídeo.
    Returns:
        np.ndarray: "Frame médio", formatado como um array numpy de inteiros positivos de 8 bits.
    """
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