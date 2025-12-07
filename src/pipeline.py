import numpy as np
import cv2 as cv
from prc.background import avg_frame
from prc.motion import MotionDetector

BIN_THRESHOLD = 30 # Limiar para binarização
MIN_AREA = 500     # Área mínima para considerar um contorno como veículo

def detect_cars(input, output, roi_polygon):
    """
    Detecta e marca veículos dentro de uma ROI poligonal num vídeo.
    
    Args:
        input: Caminho para o ficheiro de vídeo de entrada.
        output: Caminho para o ficheiro de vídeo de saída (mp4).
        roi_polygon: Array Nx2 com vértices do polígono da ROI.
    """
    background = avg_frame(input) # Calcular fundo médio
    
    detector = MotionDetector(
        background=background,
        threshold=BIN_THRESHOLD,
        min_area=MIN_AREA,
        roi_polygon=roi_polygon
    )
    
    cap = cv.VideoCapture(input)
    w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv.CAP_PROP_FPS)
    
    out = cv.VideoWriter(
        output,
        cv.VideoWriter_fourcc(*"mp4v"),
        fps, (w, h)
    )
    
    print(f"A processar: {input}")
    print(f"Resolução: {w}x{h}, FPS: {fps}")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = detector.detect_motion_regions(frame)                # Detetar regiões de movimento
        annotated_frame = detector.visualize_detection(frame, results) # Visualizar resultados
        out.write(annotated_frame)                                     # "Escrever" frame processado
        cv.imshow("Car Detection", annotated_frame)                    # Mostrar preview
        
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
    print(f"Vídeo salvo em: {output}")