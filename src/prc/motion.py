"""
Módulo para detecção de movimento e processamento de regiões ativas.
Inclui subtração de fundo, limiarização e operações morfológicas.
"""
import cv2 as cv
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt

class MotionDetector:
    """Classe para detecção e processamento de movimento."""
    def __init__(self, background, threshold, min_area, roi_polygon):
        """
        Inicializa o detetor de movimento.
        
        Args:
            background (np.ndarray):   Imagem de fundo (se None, será calculada).
            threshold (int) :          Limiar para binarização.
            min_area (int) :           Área mínima para considerar uma região como veículo.
            roi_polygon (np.ndarray) : Polígono da região de interesse.
        """
        self.background = background
        self.threshold = threshold
        self.min_area = min_area
        self.roi_polygon = roi_polygon
        self._roi_mask = None
        
        self._kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5)) # Configurar kernel morfológico
        
        if roi_polygon is not None:
            self._create_roi_mask()
    
    def _create_roi_mask(self):
        """Cria máscara binária para a região de interesse."""
        h, w = self.background.shape[:2]
        
        self._roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv.fillPoly(self._roi_mask, [self.roi_polygon], 255)
    
    def subtract_background(self, frame):
        """
        Subtrai o fundo do frame atual.
        Args:
            frame (np.ndarray) : Frame atual (BGR).
        Returns:
            np.ndarray : Diferença absoluta em escala de cinza.
        """
        diff = cv.absdiff(frame, self.background)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        
        return gray
    
    def apply_roi_mask(self, image):
        """
        Aplica máscara de ROI à imagem.
        Args:
            image (np.ndarray) : Imagem de entrada.
        Returns:
            np.ndarray : Imagem filtrada pela ROI.
        """        
        if len(image.shape) == 3:
            # Para imagens coloridas, aplicar máscara a cada canal
            masked = cv.bitwise_and(image, image, mask=self._roi_mask)
        else:
            # Para imagens em escala de cinza
            masked = cv.bitwise_and(image, self._roi_mask)
        
        return masked
    
    def threshold_image(self, image, threshold = None):
        """
        Binariza uma imagem com base num limiar.
        Args:
            image (np.ndarray) : Imagem de entrada (greyscale).
            threshold (int) : Valor do limiar (usa 'self.threshold' se None).
        Returns:
            np.ndarray : Imagem binarizada.
        """
        if threshold is None:
            threshold = self.threshold

        _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return binary
    
    def apply_morphology(self, binary, operations):
        """
        Aplica operações morfológicas à imagem binarizada.
        Args:
            binary (np.ndarray) : Imagem binarizada.
            operations (List[str]) : Lista de operações a aplicar.
        Returns:
            np.ndarray : Imagem processada.
        """
        if operations is None:
            operations = ['close', 'open']
        
        processed = binary.copy()
        
        for op in operations:
            if op == 'close':
                # Fechamento (preenche buracos)
                processed = cv.morphologyEx(processed, cv.MORPH_CLOSE, self._kernel, iterations=2)
            elif op == 'open':
                # Abertura (remove ruído)
                processed = cv.morphologyEx(processed, cv.MORPH_OPEN, self._kernel, iterations=2)
            elif op == 'dilate':
                # Dilatação
                processed = cv.dilate(processed, self._kernel, iterations=1)
            elif op == 'erode':
                # Erosão
                processed = cv.erode(processed, self._kernel, iterations=1)
        
        return processed
    
    def find_contours(self, binary, min_area):
        """
        Encontra contornos na imagem binarizada.
        Args:
            binary (np.ndarray) : Imagem binarizada.
            min_area (int): Área mínima para filtrar contornos.
        Returns:
            List[np.ndarray] : Lista de contornos válidos.
        """
        if min_area is None:
            min_area = self.min_area
            
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        valid_contours = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area >= min_area:
                valid_contours.append(cnt)
        
        return valid_contours
    
    def get_bounding_boxes(self, contours):
        """
        Extrai bounding boxes dos contornos.
        
        Args:
            contours (List[np.ndarray]) : Lista de contornos.
            
        Returns:
            List[Tuple[int, int, int, int]] : Lista de bounding boxes (x, y, largura, altura).
        """
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_motion_regions(self, frame):
        """
        Pipeline completo de detecção de regiões de movimento.
        Inclui melhorias para detetar carros pequenos (distantes).
        
        Args:
            frame (np.ndarray) : Frame atual.
        Returns:
            Dict[str, Any] : Dicionário com resultados do processamento.
        """
        results = {}
        
        # 1. Subtrair fundo
        diff_gray = self.subtract_background(frame)
        results['diff_gray'] = diff_gray
        
        # 2. Aplicar máscara ROI
        masked = self.apply_roi_mask(diff_gray)
        results['masked'] = masked
        
        # 3. MELHORIA: Aplicar CLAHE para melhorar contraste em carros distantes
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(masked)
        results['enhanced'] = enhanced
        
        # 4. MELHORIA: Limiarização adaptativa (melhor para variações de tamanho/iluminação)
        binary = cv.adaptiveThreshold(
            enhanced, 
            255, 
            cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY, 
            11,  # Tamanho do bloco
            2    # Constante subtraída
        )
        results['binary'] = binary
        
        # 5. MELHORIA: Operações morfológicas otimizadas para carros pequenos
        # 5.1 Primeiro: dilatar levemente para juntar regiões fragmentadas
        kernel_dilate = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        dilated = cv.dilate(binary, kernel_dilate, iterations=1)
        
        # 5.2 Segundo: fechamento para preencher buracos em carros maiores
        kernel_close = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        closed = cv.morphologyEx(dilated, cv.MORPH_CLOSE, kernel_close)
        
        # 5.3 Terceiro: abertura leve para remover ruído pequeno
        kernel_open = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        processed = cv.morphologyEx(closed, cv.MORPH_OPEN, kernel_open, iterations=1)
        
        results['processed'] = processed
        
        # 6. Encontrar contornos
        contours = self.find_contours(processed)
        results['contours'] = contours
        
        # 7. Extrair bounding boxes
        bboxes = self.get_bounding_boxes(contours)
        results['bboxes'] = bboxes
        
        # 8. Calcular centróides
        centroids = []
        for cnt in contours:
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        
        results['centroids'] = centroids
        
        # 9. MELHORIA: Calcular áreas para diagnóstico
        areas = [cv.contourArea(cnt) for cnt in contours]
        results['areas'] = areas
        
        # 10. MELHORIA: Filtrar por razão de aspecto (carros são mais largos que altos)
        filtered_bboxes = []
        filtered_contours = []
        filtered_centroids = []
        
        for i, (bbox, cnt, centroid) in enumerate(zip(bboxes, contours, centroids)):
            x, y, w, h = bbox
            
            # Razão de aspecto típica de carros (pelo menos 1.2 vezes mais largo que alto)
            aspect_ratio = w / h if h > 0 else 0
            
            # Filtrar por área mínima E razão de aspecto razoável
            if areas[i] >= self.min_area and aspect_ratio >= 0.8:  # Limite mais flexível
                filtered_bboxes.append(bbox)
                filtered_contours.append(cnt)
                filtered_centroids.append(centroid)
        
        # Atualizar resultados filtrados
        results['bboxes'] = filtered_bboxes
        results['contours'] = filtered_contours
        results['centroids'] = filtered_centroids
        results['aspect_ratios'] = [w/h for x,y,w,h in filtered_bboxes if h > 0]
        
        return results
    
    def visualize_detection(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Visualiza os resultados da detecção no frame original.
        Args:
            frame (np.ndarray) : Frame original.
            results (Dict[str, Any]) : Dicionário com resultados do processamento.
        Returns:
            np.ndarray : Frame com anotações.
        """
        annotated = frame.copy()
        
        # Desenhar polígono ROI
        if self.roi_polygon is not None:
            cv.polylines(annotated, [self.roi_polygon], True, (0, 255, 0), 2)
        
        # Desenhar bounding boxes
        for bbox in results.get('bboxes', []):
            x, y, w, h = bbox
            cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Desenhar centróides
        for centroid in results.get('centroids', []):
            cx, cy = centroid
            cv.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        
        return annotated