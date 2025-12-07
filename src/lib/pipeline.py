from lib.background import background
from lib.motion import MotionDetector
from lib.tracker import VehicleTracker
import cv2 as cv
import numpy as np
from typing import Tuple, Optional

class Pipeline:
    def __init__(self, input_path, output_path, roi_polygon = None, m_per_px = 0.02, min_area = 500):
        """
        Args:
            input_path (str): Diretoria do ficheiro de vídeo.
            output_path (str): Diretoria do ficheiro de vídeo de saída.
            roi_polygon (Optional[np.ndarray]): Polígono que define a ROI. Se None, a ROI não será utilizada.
            scale_m_per_px (float): Escala de "metros por pixel" do vídeo. Defaults to 0.02.
            min_area (int): Área mínima de um objeto para ser considerado um carro. Defaults to 500.
        """
        self.input_path = input_path
        self.output_path = output_path
        self.roi_polygon = roi_polygon
        self.m_per_px = m_per_px
        self.min_area = min_area

        self.background = None
        self.detector = None
        self.tracker = VehicleTracker(iou_threshold=0.25, max_missing=8)

    def _compute_roi_bbox(self, polygon, frame_shape):
        """
        Calcula a bounding box de uma ROI definida por um polígono.
        Args:
            polygon (np.ndarray): Polígono que define a ROI, formato Nx2.
            frame_shape (Tuple[int,int]): Dimensões do frame, formato (height, width).
        Returns:
            Tuple[int,int,int,int]: Bounding box da ROI, formato (x1, y1, width, height).
        """
        xs = polygon[:,0]
        ys = polygon[:,1]
        x1 = int(max(0, xs.min()))
        y1 = int(max(0, ys.min()))
        x2 = int(min(frame_shape[1]-1, xs.max()))
        y2 = int(min(frame_shape[0]-1, ys.max()))
        return (x1, y1, x2-x1, y2-y1)

    def run(self):
        """
        Executa o pipeline de detecção de veículos.

        O pipeline é composto por 3 etapas:

        1. Background: constrói uma imagem de fundo para o vídeo.
        2. ROI bbox: calcula uma bounding box para a ROI, se fornecida.
        3. Video writer: escreve o vídeo processado em um arquivo.

        Durante a execução, o pipeline também exibe uma prévia do vídeo processado em uma janela.

        Raises:
            RuntimeError: se o vídeo de entrada não puder ser aberto.
        """
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
        bg = background(self.input_path)
        self.background = bg
        self.detector = MotionDetector(background=self.background, min_area=self.min_area)

        # 2) ROI bbox
        roi_bbox = None
        roi_mask_full = None
        if self.roi_polygon is not None:
            roi_bbox = self._compute_roi_bbox(self.roi_polygon, (h,w))
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

            self.tracker.update(detections_full, frame_id, fps, self.m_per_px)

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