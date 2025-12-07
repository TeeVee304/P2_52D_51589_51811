from typing import List, Tuple, Dict
import cv2 as cv
from lib.vehicle import Vehicle
import numpy as np
from utils import iou, centroid_from_bbox
import logging

class VehicleTracker:
    """Associa detecções a tracks usando IOU greedy matching."""
    def __init__(self, iou_threshold = 0.3, max_missing = 5):
        """
        Args:
            iou_threshold (float): IOU threshold for matching detections to tracks. Defaults to 0.3.
            max_missing (int): Maximum number of frames a track can be missing before being removed. Defaults to 5.
        """
        self.tracks: Dict[str, Vehicle] = {}
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing

    def update(self, detections, frame_id, fps, scale_m_per_px):
        """
        Atualiza as tracks com base nas detecções fornecidas.
        Args:
            detections (List[Tuple[int,int,int,int]]) : Lista de bounding boxes (x,y,w,h) das detecções.
            frame_id (int) : Número de frame do frame atual.
            fps (float) : Frames por segundo do vídeo.
            scale_m_per_px (float) : Escala de "metros por pixel" do vídeo.
        """
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
            logging.debug(f"Removing track {tid} after {self.tracks[tid].missing_frames} missing frames")
            del self.tracks[tid]

        # Create new tracks for unmatched detections
        for di, bbox in enumerate(detections):
            if di in used_dets:
                continue
            centroid = det_centroids[di]
            new_v = Vehicle(bbox=bbox, centroid=centroid, frame_id=frame_id)
            new_v.compute_speed(centroid, fps, scale_m_per_px)
            self.tracks[new_v.id] = new_v