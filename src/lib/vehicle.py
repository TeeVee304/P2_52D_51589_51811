from dataclasses import dataclass, field
from collections import deque
from typing import Tuple, Deque
import uuid
import math
import numpy as np

@dataclass
class Vehicle:
    bbox: Tuple[int,int,int,int]
    centroid: Tuple[int,int]
    frame_id: int

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trail: Deque[Tuple[int,int]] = field(default_factory=lambda: deque(maxlen=20))
    first_seen: int = field(init=False)
    last_seen: int = field(init=False)
    missing_frames: int = 0
    speed_history: Deque[float] = field(default_factory=lambda: deque(maxlen=5))
    estimated_speed: float = 0.0
    color: Tuple[int,int,int] = field(default_factory=lambda: (
        int(np.random.randint(0,255)), int(np.random.randint(0,255)), int(np.random.randint(0,255))
    ))

    def __post_init__(self):
        self.trail.append(self.centroid)
        self.first_seen = self.frame_id
        self.last_seen = self.frame_id

    def update(self, bbox, centroid, frame_id):
        self.bbox = bbox
        self.centroid = centroid
        self.trail.append(centroid)
        self.last_seen = frame_id
        self.missing_frames = 0

    def mark_missed(self):
        self.missing_frames += 1

    def compute_speed(self, new_centroid, fps, scale_m_per_px):
        if len(self.trail) < 1:
            return 0.0
        prev = self.trail[-1]
        dx = new_centroid[0] - prev[0]
        dy = new_centroid[1] - prev[1]
        dist_px = math.hypot(dx, dy)
        dist_m = dist_px * scale_m_per_px
        # time between frames is 1/fps
        if fps <= 0:
            return 0.0
        speed_mps = dist_m * fps
        speed_kmh = speed_mps * 3.6
        self.speed_history.append(speed_kmh)
        self.estimated_speed = float(sum(self.speed_history)/len(self.speed_history))
        return self.estimated_speed