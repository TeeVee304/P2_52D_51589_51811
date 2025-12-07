from dataclasses import dataclass, field
from collections import deque
from typing import Tuple, Deque
import uuid
import math

@dataclass
class Vehicle:
    bbox: Tuple[int, int, int, int]     # (x, y, w, h)
    centroid: Tuple[int, int]           # (cx, cy)
    frame_id: int                       # frame de entrada

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trail: Deque[Tuple[int, int]] = field(default_factory=lambda: deque(maxlen=10))
    speed_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    speed_kmh: float = 0.0              # velocidade suavizada

    def __post_init__(self):
        self.trail.append(self.centroid)

    def update(self, bbox: Tuple[int,int,int,int], centroid: Tuple[int,int], fps: float, scale_m_per_px: float):
        """
        Atualiza estado do veículo + calcula velocidade.
        """
        self._update_speed(centroid, fps, scale_m_per_px)

        self.bbox = bbox
        self.centroid = centroid
        self.trail.append(centroid)

    def _update_speed(self, new_centroid, fps, scale):
        """
        Calcula a velocidade entre o último centróide e o novo.
        """
        if len(self.trail) == 0 or fps <= 0:
            return 0.0

        prev = self.trail[-1]
        dx = new_centroid[0] - prev[0]
        dy = new_centroid[1] - prev[1]

        dist_px = math.hypot(dx, dy)
        dist_m = dist_px * scale

        # velocidade instantânea
        speed_mps = dist_m * fps
        speed_kmh = speed_mps * 3.6

        self.speed_buffer.append(speed_kmh)
        self.speed_kmh = sum(self.speed_buffer) / len(self.speed_buffer)

        return self.speed_kmh