"""
Modular vehicle detection & tracking pipeline

Arquitetura:
- BackgroundExtractor: cria fundo usando mediana ou MOG2 (opcional)
- MotionDetector: detecção eficiente limitada à ROI (recorte antes de processar)
- Vehicle: estrutura de dados para cada veículo (baseada no seu dataclass)
- VehicleTracker: associação entre detecções e tracks (IOU + greedy matching), remoção de tracks perdidos
- Pipeline: organiza leitura do vídeo, extração de background, detecção, rastreio, anotação e gravação

Uso:
python vehicle_pipeline.py --input input.mp4 --output out.mp4 --roi "x1,y1 x2,y2 x3,y3 ..." --mode median --scale 0.02

Notas:
- Substitui a arquitetura monolítica anterior por componentes reutilizáveis
- Matching simples por IOU (greedy) — robusto e sem dependências externas
- Background: mediana de amostra de frames por ser mais resistente a tráfego constante
- ROI: aplicado como crop antes do processamento para economizar CPU

"""

from dataclasses import dataclass, field
from typing import Deque, Tuple, List, Dict, Optional
from collections import deque
import cv2 as cv
import numpy as np
import uuid
import math
import argparse
# logging removido para Jupyter
import sys

# ------------------------ Utilitários ------------------------

def iou(boxA, boxB):
    # box: (x,y,w,h)
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]

    denom = float(boxAArea + boxBArea - interArea)
    if denom <= 0:
        return 0.0
    return interArea / denom


def centroid_from_bbox(bbox):
    x, y, w, h = bbox
    return (int(x + w/2), int(y + h/2))

# ------------------------ Background extractor ------------------------

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

# ------------------------ Motion detector ------------------------

class MotionDetector:
    """Detecção dentro de ROI com recorte prévio e pipeline parametrizável."""
    def __init__(self, background: np.ndarray, min_area: int = 500,
                 kernel_size: Tuple[int,int]=(3,3)):
        self.background = background
        self.min_area = min_area
        self.kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, kernel_size)

    def subtract_background(self, frame: np.ndarray) -> np.ndarray:
        # Assume frames BGR
        diff = cv.absdiff(frame, self.background)
        gray = cv.cvtColor(diff, cv.COLOR_BGR2GRAY)
        return gray

    def detect(self, frame: np.ndarray, roi_mask: Optional[np.ndarray]=None) -> List[Tuple[int,int,int,int]]:
        # frame is already cropped to ROI when possible
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

# ------------------------ Vehicle dataclass ------------------------

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

# ------------------------ Tracker ------------------------

class VehicleTracker:
    """Associa detecções a tracks usando IOU greedy matching.

    Limitações: método simples mas eficiente, sem dependências externas.
    """
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 5):
        self.tracks: Dict[str, Vehicle] = {}
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    def update(self, detections: List[Tuple[int,int,int,int]], frame_id: int,
               fps: float, scale_m_per_px: float):
        # detections are bboxes in the same coord space as tracks
        assigned_tracks = set()
        assigned_dets = set()

        det_centroids = [centroid_from_bbox(b) for b in detections]

        # Build list of existing track bboxes
        track_items = list(self.tracks.items())  # list of (id, Vehicle)

        # Compute IOU matrix
        iou_matrix = []
        for tid, tr in track_items:
            row = [iou(tr.bbox, d) for d in detections]
            iou_matrix.append(row)

        # Greedy matching: find best IOU pairs iteratively
        matches = []  # list of (track_idx, det_idx)
        used_tracks = set()
        used_dets = set()
        while True:
            best_val = 0.0
            best_pair = None
            for ti, row in enumerate(iou_matrix):
                if ti in used_tracks:
                    continue
                for di, val in enumerate(row):
                    if di in used_dets:
                        continue
                    if val > best_val:
                        best_val = val
                        best_pair = (ti, di)
            if best_pair is None or best_val < self.iou_threshold:
                break
            ti, di = best_pair
            used_tracks.add(ti)
            used_dets.add(di)
            matches.append((ti, di))

        # Update matched tracks
        for ti, di in matches:
            tid, tr = track_items[ti]
            bbox = detections[di]
            centroid = det_centroids[di]
            tr.update(bbox, centroid, frame_id)
            tr.compute_speed(centroid, fps, scale_m_per_px)
            assigned_tracks.add(tid)
            assigned_dets.add(di)

        # Mark unmatched tracks as missed
        unmatched_tracks = [ (tid, tr) for idx, (tid,tr) in enumerate(track_items) if idx not in used_tracks ]
        for tid, tr in unmatched_tracks:
            tr.mark_missed()

        # Remove tracks missing for too long
        to_delete = [tid for tid, tr in self.tracks.items() if tr.missing_frames > self.max_missing]
        for tid in to_delete:
            print(f"Removing track {tid} after {self.tracks[tid].missing_frames} missing frames")
            del self.tracks[tid]

        # Create new tracks for unmatched detections
        for di, bbox in enumerate(detections):
            if di in used_dets:
                continue
            centroid = det_centroids[di]
            new_v = Vehicle(bbox=bbox, centroid=centroid, frame_id=frame_id)
            new_v.compute_speed(centroid, fps, scale_m_per_px)
            self.tracks[new_v.id] = new_v

    def draw_tracks(self, frame: np.ndarray, offset=(0,0)) -> np.ndarray:
        # offset is used when working with cropped ROI to draw back in full frame coords
        annotated = frame.copy()
        ox, oy = offset
        for tid, tr in self.tracks.items():
            x,y,w,h = tr.bbox
            x += ox; y += oy
            cv.rectangle(annotated, (x,y), (x+w, y+h), tr.color, 2)
            cx, cy = tr.centroid
            cv.circle(annotated, (cx+ox, cy+oy), 3, tr.color, -1)
            cv.putText(annotated, f"{tid[:6]} {int(tr.estimated_speed)}km/h",
                       (x, y-6), cv.FONT_HERSHEY_SIMPLEX, 0.5, tr.color, 1)
            # draw trail
            pts = list(tr.trail)
            for i in range(1, len(pts)):
                cv.line(annotated, (pts[i-1][0]+ox, pts[i-1][1]+oy), (pts[i][0]+ox, pts[i][1]+oy), tr.color, 2)
        return annotated

# ------------------------ Pipeline ------------------------

class Pipeline:
    def __init__(self, input_path: str, output_path: str, roi_polygon: Optional[np.ndarray]=None,
                 scale_m_per_px: float = 0.02, min_area: int = 500):
        self.input_path = input_path
        self.output_path = output_path
        self.roi_polygon = roi_polygon
        self.scale_m_per_px = scale_m_per_px
        self.min_area = min_area

        self.bg_extractor = BackgroundExtractor()
        self.background = None
        self.detector = None
        self.tracker = VehicleTracker(iou_threshold=0.25, max_missing=8)

    def compute_roi_bbox(self, polygon: np.ndarray, frame_shape: Tuple[int,int]) -> Tuple[int,int,int,int]:
        # polygon: Nx2
        xs = polygon[:,0]
        ys = polygon[:,1]
        x1 = int(max(0, xs.min()))
        y1 = int(max(0, ys.min()))
        x2 = int(min(frame_shape[1]-1, xs.max()))
        y2 = int(min(frame_shape[0]-1, ys.max()))
        return (x1, y1, x2-x1, y2-y1)

    def run(self, mode: str = 'median'):
        cap = cv.VideoCapture(self.input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Erro a abrir {self.input_path}")
        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
            print("FPS inválido detectado; usando 25")

        # 1) Background
        print("Construindo background...")
        if mode == 'median':
            bg = self.bg_extractor.build_median_background(self.input_path)
        else:
            bg = self.bg_extractor.build_mog2_background(self.input_path)
        self.background = bg
        self.detector = MotionDetector(background=self.background, min_area=self.min_area)

        # 2) ROI bbox
        roi_bbox = None
        roi_mask_full = None
        if self.roi_polygon is not None:
            roi_bbox = self.compute_roi_bbox(self.roi_polygon, (h,w))
            # create mask for cropped ROI
            roi_mask_full = np.zeros((h,w), dtype=np.uint8)
            cv.fillPoly(roi_mask_full, [self.roi_polygon], 255)

        # 3) Video writer
        fourcc = cv.VideoWriter_fourcc(*"mp4v")
        out = cv.VideoWriter(self.output_path, fourcc, fps, (w,h))

        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1

            # Crop to ROI early to save processing
            if roi_bbox is not None:
                x,y,ww,hh = roi_bbox
                cropped = frame[y:y+hh, x:x+ww]
                # mask relative to crop
                roi_mask = roi_mask_full[y:y+hh, x:x+ww]
            else:
                cropped = frame
                roi_mask = None

            detections, binary = self.detector.detect(cropped, roi_mask)

            # convert detections coords back to full frame
            if roi_bbox is not None:
                detections_full = [(x+bx, y+by, bw, bh) for (bx,by,bw,bh) in detections]
            else:
                detections_full = detections

            self.tracker.update(detections_full, frame_id, fps, self.scale_m_per_px)

            annotated = self.tracker.draw_tracks(frame, offset=(0,0))

            # optional: draw ROI
            if self.roi_polygon is not None:
                cv.polylines(annotated, [self.roi_polygon], True, (255,0,0), 2)

            out.write(annotated)

            # show preview (non-blocking)
            cv.imshow('Pipeline', annotated)
            if cv.waitKey(1) & 0xFF == 27:
                print('Interrompido pelo usuário')
                break

        cap.release()
        out.release()
        cv.destroyAllWindows()
        print('Processamento terminado')

# ------------------------ CLI ------------------------

def parse_roi(s: str) -> np.ndarray:
    # string like: "x1,y1 x2,y2 x3,y3"
    parts = s.strip().split()
    pts = []
    for p in parts:
        x,y = p.split(',')
        pts.append([int(x), int(y)])
    return np.array(pts, dtype=np.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--roi', required=False, help='"x1,y1 x2,y2 ..."')
    parser.add_argument('--mode', choices=['median','mog2'], default='median')
    parser.add_argument('--scale', type=float, default=0.02, help='metros por pixel')
    parser.add_argument('--min_area', type=int, default=500)
    args = parser.parse_args()

    print(level=print, format='%(asctime)s %(levelname)s %(message)s')

    if args.roi:
        roi_poly = parse_roi(args.roi)
    else:
        roi_poly = None

    pipe = Pipeline(input_path=args.input, output_path=args.output, roi_polygon=roi_poly,
                    scale_m_per_px=args.scale, min_area=args.min_area)
    pipe.run(mode=args.mode)

if __name__ == '__main__':
    main()
