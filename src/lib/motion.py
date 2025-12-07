import cv2 as cv
import numpy as np

class MotionDetector:
    def __init__(self, background, threshold=30, min_area=500, roi_polygon=None):
        self.background = background
        self.threshold = threshold
        self.min_area = min_area
        self.roi_polygon = roi_polygon
        self._roi_mask = None

        if roi_polygon is not None:
            h, w = background.shape[:2]
            self._roi_mask = np.zeros((h, w), dtype=np.uint8)
            cv.fillPoly(self._roi_mask, [roi_polygon], 255)

    def _subtract_background(self, frame):
        """
        Subtrai o fundo (background) de um frame, aplicando uma máscara de ROI caso seja fornecida.
        Args:
            frame (np.ndarray): Frame a ser processado.
        Returns:
            np.ndarray: Imagem grayscale resultante da subtração do fundo.
        """
        diff = cv.absdiff(frame, self.background)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        if self._roi_mask is not None:
            gray = cv.bitwise_and(gray, self._roi_mask)
        return gray

    def _threshold_image(self, gray):
        """
        Converte uma imagem grayscale para uma imagem binária.
        Args:
            gray (numpy.ndarray): Imagem em grayscale.
        Returns:
            numpy.ndarray: Imagem binária.
        """
        _, binary = cv.threshold(gray, self.threshold, 255, cv.THRESH_BINARY)
        return binary

    def _find_contours(self, binary):
        """
        Encontra contornos em uma imagem binária e retorna apenas os que tem área maior ou igual a um valor mínimo.
        Args:
            binary (numpy.ndarray): Imagem binária.
        Returns:
            List[numpy.ndarray]: Lista de contornos válidos.
        """
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv.contourArea(cnt) >= self.min_area]
        return valid_contours

    def _get_bounding_boxes(self, contours):
        """
        Converte contornos em bounding boxes.
        Args:
            contours (List[numpy.ndarray]): Lista de contornos.
        Returns:
            List[Tuple[int,int,int,int]]: Lista de bounding boxes (x,y,w,h).
        """
        return [cv.boundingRect(cnt) for cnt in contours]

    def detect(self, frame):
        """
        Detecta regiões de movimento em um único frame.
        Retorna contornos e bounding boxes.
        """
        gray = self._subtract_background(frame)
        binary = self._threshold_image(gray)
        contours = self._find_contours(binary)
        bboxes = self._get_bounding_boxes(contours)
        return {'contours': contours, 'bboxes': bboxes, 'binary': binary}