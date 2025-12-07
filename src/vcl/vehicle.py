from dataclasses import dataclass, field
from typing import Deque, Tuple
import uuid
from collections import deque
import math
import numpy as np

@dataclass
class Vehicle:
    """Classe para representar um veículo rastreado."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    centroid: Tuple[int, int]        # (cx, cy)
    frame_id: int                    # Frame de primeira deteção
    
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trail: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))
    first_seen: int = field(init=False)
    last_seen: int = field(init=False)
    speed_history: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    current_speed: float = 0.0
    estimated_speed: float = 0.0
    color: Tuple[int, int, int] = field(
        default_factory=lambda: (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255)
        )
    )
    
    def __post_init__(self):
        """Inicializar atributos que dependem de outros."""
        self.trail.append(self.centroid)
        self.first_seen = self.frame_id
        self.last_seen = self.frame_id
    
    def update(self, bbox, centroid, frame_id):
        """
        Atualiza o estado do veículo.
        Args:
            bbox (Tuple[int, int, int, int]) : Nova bounding box.
            centroid (Tuple[int, int]) : Novo centróide.
            frame_id (int) : ID do frame atual.
        """
        self.bbox = bbox
        self.centroid = centroid
        self.trail.append(centroid)
        self.last_seen = frame_id
    
    def speed(self, new_centroid, fps, scale_factor):
        """
        Calcula velocidade baseada no deslocamento do centróide.
        Args:
            new_centroid (Tuple[int, int]) : Novo centróide.
            fps (float) : Frames por segundo do vídeo.
            scale_factor (float) : Metros por pixel (calibração).
        Returns:
            float : Velocidade estimada em km/h.
        """
        if len(self.trail) < 2:
            return 0.0
        
        # Obter posições atual e anterior
        prev_centroid = self.trail[-2] if len(self.trail) > 1 else self.trail[0]
        curr_centroid = new_centroid
        
        # Calcular distância em pixels
        dx = curr_centroid[0] - prev_centroid[0]
        dy = curr_centroid[1] - prev_centroid[1]
        distance_pixels = math.sqrt(dx**2 + dy**2)
        
        # Converter para unidades reais
        distance_meters = distance_pixels * scale_factor
        
        # Tempo entre frames
        time_seconds = 1.0 / fps
        
        # Velocidade em m/s e converter para km/h
        speed_mps = distance_meters / time_seconds
        speed_kmh = speed_mps * 3.6
        
        # Atualizar histórico
        self.speed_history.append(speed_kmh)
        self.current_speed = speed_kmh
        
        # Calcular velocidade média (suavizada)
        if self.speed_history:
            self.estimated_speed = sum(self.speed_history) / len(self.speed_history)
        
        return self.estimated_speed