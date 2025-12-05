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
    
    def __init__(self, 
                 background: Optional[np.ndarray] = None,
                 threshold: int = 30,
                 min_area: int = 500,
                 roi_polygon: Optional[np.ndarray] = None):
        """
        Inicializa o detetor de movimento.
        
        Args:
            background: Imagem de fundo (se None, será calculada).
            threshold: Limiar para binarização.
            min_area: Área mínima para considerar uma região como veículo.
            roi_polygon: Polígono da região de interesse.
        """
        self.background = background
        self.threshold = threshold
        self.min_area = min_area
        self.roi_polygon = roi_polygon
        self.roi_mask = None
        
        # Configurar kernel morfológico
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        
        # Inicializar máscara ROI se polígono fornecido
        if roi_polygon is not None:
            self._create_roi_mask()
    
    def _create_roi_mask(self, frame_shape: Optional[Tuple[int, int]] = None):
        """Cria máscara binária para a região de interesse."""
        if self.roi_polygon is None:
            self.roi_mask = None
            return
            
        if frame_shape is None:
            if self.background is not None:
                h, w = self.background.shape[:2]
            else:
                raise ValueError("Forneça frame_shape ou background para criar máscara ROI.")
        else:
            h, w = frame_shape[:2] if len(frame_shape) == 3 else frame_shape
        
        self.roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv.fillPoly(self.roi_mask, [self.roi_polygon], 255)
    
    def set_roi_polygon(self, polygon: np.ndarray, frame_shape: Optional[Tuple[int, int]] = None):
        """
        Define um novo polígono de ROI.
        
        Args:
            polygon: Array Nx2 com vértices do polígono.
            frame_shape: Formato do frame (altura, largura).
        """
        self.roi_polygon = polygon.astype(np.int32)
        self._create_roi_mask(frame_shape)
    
    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        """
        Subtrai o fundo do frame atual.
        
        Args:
            frame: Frame atual (BGR).
            
        Returns:
            Diferença absoluta em escala de cinza.
        """
        if self.background is None:
            raise ValueError("Background não definido. Chame set_background() primeiro.")
        
        # Calcular diferença absoluta
        diff = cv.absdiff(frame, self.background)
        
        # Converter para escala de cinza
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        
        return gray
    
    def apply_roi_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica máscara de ROI à imagem.
        
        Args:
            image: Imagem de entrada.
            
        Returns:
            Imagem com ROI aplicada.
        """
        if self.roi_mask is None:
            return image
        
        # Garantir que a máscara tem as mesmas dimensões
        if len(image.shape) == 3:
            # Para imagens coloridas, aplicar máscara a cada canal
            masked = cv.bitwise_and(image, image, mask=self.roi_mask)
        else:
            # Para imagens em escala de cinza
            masked = cv.bitwise_and(image, self.roi_mask)
        
        return masked
    
    def threshold_image(self, image: np.ndarray, 
                       threshold: Optional[int] = None,
                       method: str = 'binary') -> np.ndarray:
        """
        Aplica limiarização à imagem.
        
        Args:
            image: Imagem de entrada (escala de cinza).
            threshold: Valor do limiar (usa self.threshold se None).
            method: Método de limiarização ('binary', 'otsu', 'adaptive').
            
        Returns:
            Imagem binarizada.
        """
        if threshold is None:
            threshold = self.threshold
            
        if method == 'binary':
            _, binary = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
        elif method == 'otsu':
            # Limiarização de Otsu (automática)
            _, binary = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        elif method == 'adaptive':
            # Limiarização adaptativa
            binary = cv.adaptiveThreshold(image, 255, 
                                         cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv.THRESH_BINARY, 11, 2)
        else:
            raise ValueError(f"Método '{method}' não suportado.")
            
        return binary
    
    def apply_morphology(self, binary: np.ndarray, 
                        operations: List[str] = None) -> np.ndarray:
        """
        Aplica operações morfológicas à imagem binarizada.
        
        Args:
            binary: Imagem binarizada.
            operations: Lista de operações a aplicar.
            
        Returns:
            Imagem processada.
        """
        if operations is None:
            operations = ['close', 'open']
        
        processed = binary.copy()
        
        for op in operations:
            if op == 'close':
                # Fechamento (preenche buracos)
                processed = cv.morphologyEx(processed, cv.MORPH_CLOSE, self.kernel, iterations=2)
            elif op == 'open':
                # Abertura (remove ruído)
                processed = cv.morphologyEx(processed, cv.MORPH_OPEN, self.kernel, iterations=2)
            elif op == 'dilate':
                # Dilatação
                processed = cv.dilate(processed, self.kernel, iterations=1)
            elif op == 'erode':
                # Erosão
                processed = cv.erode(processed, self.kernel, iterations=1)
        
        return processed
    
    def find_contours(self, binary: np.ndarray, 
                     min_area: Optional[int] = None) -> List[np.ndarray]:
        """
        Encontra contornos na imagem binarizada.
        
        Args:
            binary: Imagem binarizada.
            min_area: Área mínima para filtrar contornos.
            
        Returns:
            Lista de contornos válidos.
        """
        if min_area is None:
            min_area = self.min_area
            
        # Encontrar todos os contornos
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        # Filtrar por área mínima
        valid_contours = []
        for cnt in contours:
            area = cv.contourArea(cnt)
            if area >= min_area:
                valid_contours.append(cnt)
        
        return valid_contours
    
    def get_bounding_boxes(self, contours: List[np.ndarray]) -> List[Tuple[int, int, int, int]]:
        """
        Extrai bounding boxes dos contornos.
        
        Args:
            contours: Lista de contornos.
            
        Returns:
            Lista de bounding boxes (x, y, largura, altura).
        """
        bboxes = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            bboxes.append((x, y, w, h))
        
        return bboxes
    
    def detect_motion_regions(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Pipeline completo de detecção de regiões de movimento.
        
        Args:
            frame: Frame atual.
            
        Returns:
            Dicionário com resultados do processamento.
        """
        results = {}
        
        # 1. Subtrair fundo
        diff_gray = self.subtract_background(frame)
        results['diff_gray'] = diff_gray
        
        # 2. Aplicar máscara ROI
        masked = self.apply_roi_mask(diff_gray)
        results['masked'] = masked
        
        # 3. Limiarização
        binary = self.threshold_image(masked, method='binary')
        results['binary'] = binary
        
        # 4. Operações morfológicas
        processed = self.apply_morphology(binary)
        results['processed'] = processed
        
        # 5. Encontrar contornos
        contours = self.find_contours(processed)
        results['contours'] = contours
        
        # 6. Extrair bounding boxes
        bboxes = self.get_bounding_boxes(contours)
        results['bboxes'] = bboxes
        
        # 7. Calcular centróides
        centroids = []
        for cnt in contours:
            M = cv.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
        
        results['centroids'] = centroids
        
        return results
    
    def set_background(self, background: np.ndarray):
        """Define a imagem de fundo."""
        self.background = background
        
        # Atualizar máscara ROI se necessário
        if self.roi_polygon is not None:
            self._create_roi_mask(background.shape[:2])
    
    def visualize_detection(self, frame: np.ndarray, results: Dict[str, Any]) -> np.ndarray:
        """
        Visualiza os resultados da detecção no frame original.
        
        Args:
            frame: Frame original.
            results: Dicionário com resultados do processamento.
            
        Returns:
            Frame com anotações.
        """
        annotated = frame.copy()
        
        # Desenhar polígono ROI
        if self.roi_polygon is not None:
            cv.polylines(annotated, [self.roi_polygon], True, (255, 0, 0), 2)
        
        # Desenhar bounding boxes
        for bbox in results.get('bboxes', []):
            x, y, w, h = bbox
            cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Desenhar centróides
        for centroid in results.get('centroids', []):
            cx, cy = centroid
            cv.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)
        
        return annotated


# Função de conveniência (mantém compatibilidade)
def detect_cars_polygon(video_path: str, output_path: str, roi_polygon: np.ndarray) -> None:
    """
    Detecta e marca veículos dentro de uma ROI poligonal num vídeo.
    
    Args:
        video_path: Caminho para o ficheiro de vídeo de entrada.
        output_path: Caminho para o ficheiro de vídeo de saída (mp4).
        roi_polygon: Array Nx2 com vértices do polígono da ROI.
    """
    # Calcular fundo
    background = avg_frame(video_path)
    
    # Inicializar detetor
    detector = MotionDetector(
        background=background,
        threshold=30,
        min_area=500,
        roi_polygon=roi_polygon
    )
    
    # Abrir vídeo
    cap = cv.VideoCapture(video_path)
    
    # Obter propriedades do vídeo
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    # Criar VideoWriter
    out = cv.VideoWriter(
        output_path,
        cv.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )
    
    print(f"Processando vídeo: {video_path}")
    print(f"Resolução: {w}x{h}, FPS: {fps}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detetar regiões de movimento
        results = detector.detect_motion_regions(frame)
        
        # Visualizar resultados
        annotated_frame = detector.visualize_detection(frame, results)
        
        # Escrever frame processado
        out.write(annotated_frame)
        
        # Mostrar preview
        cv.imshow("Car Detection", annotated_frame)
        
        # Contar frames processados
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Frames processados: {frame_count}")
        
        # Verificar tecla ESC para sair
        if cv.waitKey(1) & 0xFF == 27:
            print("Processamento interrompido pelo usuário.")
            break
    
    # Liberar recursos
    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    print(f"Processamento concluído. Total de frames: {frame_count}")
    print(f"Vídeo salvo em: {output_path}")