"""
Módulo para detecção de movimento e processamento de regiões ativas.
Inclui subtração de fundo, limiarização e operações morfológicas.
"""
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class MotionDetector:
    """
    Detecção dentro de ROI com recorte prévio e pipeline parametrizável.
    Args:
        background (np.ndarray): Frame de fundo para subtração.
        min_area (int): Área mínima para considerar um contorno como movimento.
        kernel_size (Tuple[int,int]): Tamanho do kernel para operações morfológicas.
    Returns:
        List[Tuple[int,int,int,int]]: Lista de bounding boxes (x,y,w,h) das regiões detectadas.
    """
    def __init__(self, background, min_area = 500, kernel_size = (3,3)):
        """
        Args:
            background (np.ndarray): Frame de fundo para subtração.
            min_area (int, optional): Área mínima para considerar um contorno como movimento. Defaults to 500.
            kernel_size (Tuple[int,int], optional): Tamanho do kernel para operações morfológicas. Defaults to (3,3).
        """
        self.background = background
        self.min_area = min_area
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Subtrai o fundo de um frame, retornando a imagem em tons de cinza.
        Args:
            frame (np.ndarray): Frame de entrada em BGR.
        Returns:
            np.ndarray: Imagem em tons de cinza resultante da subtração do fundo.
        """
        # Assume frames BGR
        diff = cv.absdiff(frame, self.background)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        return gray

    def detect(self, frame: np.ndarray, roi_mask = None):
        """
        Detecta movimento dentro de uma região de interesse (ROI) com recorte prévio e pipeline parametrizável.
        Args:
            frame (np.ndarray): Frame de entrada em BGR.
            roi_mask (Optional[np.ndarray]): Máscara de região de interesse (opcional).
        Returns:
            List[Tuple[int,int,int,int]]: Lista de bounding boxes (x,y,w,h) das regiões detectadas.
        """
        gray = self.subtract_background(frame)

        # CLAHE
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # adaptive threshold
        binary = cv.adaptiveThreshold(enhanced, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv.THRESH_BINARY, 11, 2)

        # morphology
        binary = cv.morphologyEx(binary, cv.MORPH_CLOSE, self.kernel, iterations=2)
        binary = cv.morphologyEx(binary, cv.MORPH_OPEN, self.kernel, iterations=1)

        # If roi_mask provided (for cropped frames mask is optional)
        if roi_mask is not None:
            binary = cv.bitwise_and(binary, roi_mask)

        # contours
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area < self.min_area:
                continue
            x,y,w,h = cv.boundingRect(cnt)
            bboxes.append((x,y,w,h))
        return bboxes, binary