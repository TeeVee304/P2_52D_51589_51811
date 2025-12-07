from vehicle import Vehicle
import math

class VehicleTracker:
    """Rastreia veículos entre frames e calcula velocidades."""
    def __init__(self, fps=30, scale_factor=0.1, max_disappeared=5, max_distance=50):
        """
        Args:
            fps: Frames por segundo do vídeo
            scale_factor: Metros por pixel (calibração)
            max_disappeared: Frames até remover veículo perdido
            max_distance: Distância máxima para matching (pixels)
        """
        self.vehicles = {}  # {id: Vehicle}
        self.fps = fps
        self.scale_factor = scale_factor
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_count = 0
        self.total_count = 0  # Contador total de veículos
        self.lane_speeds = {}  # Velocidades por faixa
        
    def match_vehicles(self, new_bboxes, new_centroids):
        """Associa detecções atuais a veículos existentes."""
        matched_vehicles = {}
        unmatched_indices = list(range(len(new_centroids)))
        
        # Se não há veículos anteriores, criar novos
        if not self.vehicles:
            for i, (bbox, centroid) in enumerate(zip(new_bboxes, new_centroids)):
                vehicle = Vehicle(bbox, centroid, self.frame_count)
                vehicle.id = f"V{self.total_count:03d}"
                self.vehicles[vehicle.id] = vehicle
                self.total_count += 1
                matched_vehicles[vehicle.id] = (bbox, centroid)
            return matched_vehicles
        
        # Calcular distâncias entre centróides existentes e novos
        distance_matrix = []
        for vehicle_id, vehicle in self.vehicles.items():
            distances = []
            for i, new_centroid in enumerate(new_centroids):
                # Distância euclidiana
                dist = math.sqrt(
                    (vehicle.centroid[0] - new_centroid[0])**2 +
                    (vehicle.centroid[1] - new_centroid[1])**2
                )
                distances.append((dist, i))
            distance_matrix.append((vehicle_id, distances))
        
        # Ordenar por menor distância e fazer matching
        for vehicle_id, distances in distance_matrix:
            distances.sort(key=lambda x: x[0])
            for dist, idx in distances:
                if dist < self.max_distance and idx in unmatched_indices:
                    # Match encontrado
                    matched_vehicles[vehicle_id] = (new_bboxes[idx], new_centroids[idx])
                    unmatched_indices.remove(idx)
                    break
        
        # Atualizar veículos existentes
        for vehicle_id, (bbox, centroid) in matched_vehicles.items():
            vehicle = self.vehicles[vehicle_id]
            old_centroid = vehicle.centroid
            vehicle.update(bbox, centroid, self.frame_count)
            # Calcular velocidade
            speed = vehicle.calculate_speed(centroid, self.fps, self.scale_factor)
            vehicle.estimated_speed = speed
        
        # Criar novos veículos para detecções não emparelhadas
        for idx in unmatched_indices:
            vehicle = Vehicle(new_bboxes[idx], new_centroids[idx], self.frame_count)
            vehicle.id = f"V{self.total_count:03d}"
            self.vehicles[vehicle.id] = vehicle
            matched_vehicles[vehicle.id] = (new_bboxes[idx], new_centroids[idx])
            self.total_count += 1
        
        # Remover veículos perdidos
        lost_vehicles = []
        for vehicle_id in list(self.vehicles.keys()):
            if vehicle_id not in matched_vehicles:
                if self.frame_count - self.vehicles[vehicle_id].last_seen > self.max_disappeared:
                    lost_vehicles.append(vehicle_id)
        
        for vehicle_id in lost_vehicles:
            del self.vehicles[vehicle_id]
        
        self.frame_count += 1
        return matched_vehicles