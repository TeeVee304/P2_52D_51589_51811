import cv2 as cv
import numpy as np

class MotionDetector:
    def __init__(self, min_area=400, roi_polygon=None):
        self.min_area = min_area
        self.background = None
        self.roi_polygon = roi_polygon
        self.roi_mask = None
        
        if roi_polygon is not None:
            self.roi_mask = np.zeros((1080, 1920), dtype=np.uint8)  # ajusta se necessário
            cv.fillPoly(self.roi_mask, [roi_polygon], 255)

    def _apply_roi(self, frame):
        """
        Aplica uma máscara de ROI a um frame.
        Se self.roi_mask for None, retorna o frame original.
        Caso contrário, aplica a máscara de ROI no frame e retorna o resultado.
        Args:
            frame (np.ndarray): Frame a ser processado.
        Returns:
            np.ndarray: Frame processado com a máscara de ROI aplicada.
        """
        if self.roi_mask is None:
            return frame
        return cv.bitwise_and(frame, frame, mask=self.roi_mask)
    
    def _init_background(self, frame_proc):
        """
        Inicializa o fundo (background) com base em um frame processado.
        O fundo é uma cópia do frame processado, convertida para um array numpy de inteiros positivos de 16 bits.
        Args:
            frame_proc (np.ndarray): Frame processado com base na média de todos os frames de um vídeo.
        """
        self.background = frame_proc.copy().astype(np.int16)

    def _subtract_background(self, frame_proc):
        """
        Subtrai o fundo (background) de um frame processado.
        Retorna uma imagem grayscale resultante da subtração do fundo.
        Args:
            frame_proc (np.ndarray): Frame a ser processado.
        Returns:
            np.ndarray: Imagem grayscale resultante da subtração do fundo.
        """
        diff = cv.absdiff(frame_proc, self.background.astype(np.uint8))
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        return gray

    def _threshold(self, gray):
        """
        Converte uma imagem grayscale numa imagem binária utilizando um limiar de threshold.
        O limiar de threshold é 30.
        Retorna a imagem binária resultante da conversão.
        Args:
            gray (np.ndarray): Imagem grayscale a ser convertida.
        Returns:
            np.ndarray: Imagem binária resultante da conversão.
        """
        _, binary = cv.threshold(gray, 30, 255, cv.THRESH_BINARY)
        return binary

    def _extract(self, binary):
        """
        Extraia contornos e bounding boxes de uma imagem binária.
        Args:
            binary (np.ndarray): Imagem binária a ser extraída.
        Returns:
            Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]: Tupla que contém a lista de contornos e lista de bounding boxes.
        """
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        final_contours = []

        for c in contours:
            area = cv.contourArea(c)
            if area < self.min_area:
                continue

            x, y, w, h = cv.boundingRect(c)
            bboxes.append((x, y, w, h))
            final_contours.append(c)

        return final_contours, bboxes

    def detect(self, frame):
        """
        Detecta regiões de movimento num único frame.
        Retorna contornos e bounding boxes.
        """
        frame_proc = self._apply_roi(frame)

        # inicializa background no MESMO TAMANHO e FORMATO do frame_proc
        if self.background is None:
            self._init_background(frame_proc)

        gray = self._subtract_background(frame_proc)
        binary = self._threshold(gray)
        contours, bboxes = self._extract(binary)

        return {
            "contours": contours,
            "bboxes": bboxes,
            "binary": binary
        }
