"""
ADS-B Data Processing Layer

Processes aircraft data from ADS-B.lol API (within 3 nautical miles of airports)
to identify bottleneck zones and construct operational graphs.
"""

import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import torch
from torch_geometric.data import Data
import geopy.distance


@dataclass
class AircraftData:
    """Individual aircraft data structure"""
    flight_id: str
    aircraft_type: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    speed: float
    phase: str  # approach, departure, taxi, gate, holding
    timestamp: str


@dataclass
class BottleneckZone:
    """Bottleneck zone definition"""
    zone_id: str
    zone_type: str  # runway_approach, runway_departure, taxiway_intersection, gate_area
    center_lat: float
    center_lon: float
    radius_meters: float
    capacity: int
    current_load: int
    bottleneck_probability: float


class ADSBDataProcessor:
    """
    Processes ADS-B data to identify bottleneck zones and construct spatial graphs
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.spatial_resolution = config.get('spatial_resolution_meters', 100)
        self.bottleneck_threshold = config.get('bottleneck_detection_threshold', 0.6)
        
    def load_adsb_data(self, json_file_path: str) -> Dict:
        """
        Load ADS-B data from JSON file
        
        Args:
            json_file_path: Path to ADS-B.lol API response JSON
            
        Returns:
            Dictionary containing aircraft data
        """
        try:
            with open(json_file_path, 'r') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading ADS-B data: {e}")
            return {}
    
    def filter_airport_operations(self, data: Dict, airport_icao: str) -> List[AircraftData]:
        """
        Filter aircraft data for operations within airport radius
        
        Args:
            data: Raw ADS-B data
            airport_icao: Airport ICAO code (e.g., 'KJFK')
            
        Returns:
            List of AircraftData objects within airport operations area
        """
        aircraft_list = []
        
        # Get airport coordinates (simplified - in production, use airport database)
        airport_coords = self._get_airport_coordinates(airport_icao)
        if not airport_coords:
            return aircraft_list
            
        airport_lat, airport_lon = airport_coords
        radius_nm = self.config.get('adsb_radius_nm', 3)
        
        for aircraft in data.get('aircraft', []):
            try:
                lat = aircraft.get('lat')
                lon = aircraft.get('lon')
                
                if lat is None or lon is None:
                    continue
                    
                # Calculate distance from airport
                distance_nm = geopy.distance.distance(
                    (airport_lat, airport_lon), 
                    (lat, lon)
                ).nautical
                
                if distance_nm <= radius_nm:
                    aircraft_data = AircraftData(
                        flight_id=aircraft.get('flight', 'UNKNOWN'),
                        aircraft_type=aircraft.get('t', 'UNKNOWN'),
                        latitude=lat,
                        longitude=lon,
                        altitude=aircraft.get('alt_baro', 0),
                        heading=aircraft.get('track', 0),
                        speed=aircraft.get('gs', 0),
                        phase=self._determine_aircraft_phase(aircraft),
                        timestamp=aircraft.get('timestamp', '')
                    )
                    aircraft_list.append(aircraft_data)
                    
            except Exception as e:
                print(f"Error processing aircraft data: {e}")
                continue
                
        return aircraft_list
    
    def identify_bottleneck_zones(self, flights: List[AircraftData]) -> List[BottleneckZone]:
        """
        Identify potential bottleneck zones based on aircraft density and patterns
        
        Args:
            flights: List of aircraft in airport area
            
        Returns:
            List of identified bottleneck zones
        """
        bottleneck_zones = []
        
        # Define airport bottleneck zones based on typical airport layout
        zones = self._define_airport_zones()
        
        for zone_config in zones:
            zone_aircraft = self._get_aircraft_in_zone(flights, zone_config)
            
            if len(zone_aircraft) > 0:
                bottleneck_prob = self._calculate_bottleneck_probability(
                    zone_aircraft, zone_config
                )
                
                if bottleneck_prob > self.bottleneck_threshold:
                    bottleneck_zone = BottleneckZone(
                        zone_id=zone_config['zone_id'],
                        zone_type=zone_config['zone_type'],
                        center_lat=zone_config['center_lat'],
                        center_lon=zone_config['center_lon'],
                        radius_meters=zone_config['radius_meters'],
                        capacity=zone_config['capacity'],
                        current_load=len(zone_aircraft),
                        bottleneck_probability=bottleneck_prob
                    )
                    bottleneck_zones.append(bottleneck_zone)
                    
        return bottleneck_zones
    
    def construct_bottleneck_graph(self, flights: List[AircraftData], 
                                 airport_config: Dict) -> Data:
        """
        Construct PyTorch Geometric graph for bottleneck analysis
        
        Args:
            flights: List of aircraft data
            airport_config: Airport configuration data
            
        Returns:
            PyTorch Geometric Data object representing the spatial graph
        """
        # Create spatial grid for graph construction
        grid_size = self.spatial_resolution
        airport_lat, airport_lon = self._get_airport_coordinates(
            airport_config.get('icao', 'KJFK')
        )
        
        # Define grid bounds (simplified)
        grid_bounds = {
            'min_lat': airport_lat - 0.01,  # ~1km
            'max_lat': airport_lat + 0.01,
            'min_lon': airport_lon - 0.01,
            'max_lon': airport_lon + 0.01
        }
        
        # Create grid nodes
        nodes = []
        node_features = []
        
        for i, flight in enumerate(flights):
            # Convert lat/lon to grid coordinates
            grid_x = int((flight.longitude - grid_bounds['min_lon']) / 
                        (grid_bounds['max_lon'] - grid_bounds['min_lon']) * grid_size)
            grid_y = int((flight.latitude - grid_bounds['min_lat']) / 
                        (grid_bounds['max_lat'] - grid_bounds['min_lat']) * grid_size)
            
            nodes.append([grid_x, grid_y])
            
            # Create node features: [aircraft_type_encoded, altitude, speed, heading, phase_encoded]
            features = [
                self._encode_aircraft_type(flight.aircraft_type),
                flight.altitude / 1000.0,  # Normalize altitude
                flight.speed / 100.0,     # Normalize speed
                flight.heading / 360.0,   # Normalize heading
                self._encode_phase(flight.phase)
            ]
            node_features.append(features)
        
        # Create edges based on spatial proximity
        edges = []
        edge_weights = []
        
        for i in range(len(flights)):
            for j in range(i + 1, len(flights)):
                # Calculate distance between aircraft
                dist = geopy.distance.distance(
                    (flights[i].latitude, flights[i].longitude),
                    (flights[j].latitude, flights[j].longitude)
                ).meters
                
                # Create edge if aircraft are within interaction distance
                if dist < 2000:  # 2km interaction radius
                    edges.append([i, j])
                    edges.append([j, i])  # Undirected graph
                    weight = 1.0 / (dist / 1000.0 + 1.0)  # Inverse distance weight
                    edge_weights.extend([weight, weight])
        
        # Convert to PyTorch tensors
        if len(node_features) == 0:
            # Empty graph fallback
            node_features = torch.zeros((1, 5))
            edges = torch.zeros((2, 0), dtype=torch.long)
            edge_weights = torch.zeros(0)
        else:
            node_features = torch.tensor(node_features, dtype=torch.float32)
            edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
            edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        
        return Data(
            x=node_features,
            edge_index=edges,
            edge_attr=edge_weights,
            num_nodes=len(node_features)
        )
    
    def _get_airport_coordinates(self, airport_icao: str) -> Optional[Tuple[float, float]]:
        """Get airport coordinates (simplified implementation)"""
        # In production, use proper airport database
        airport_coords = {
            'KJFK': (40.6413, -73.7781),
            'KLAX': (33.9425, -118.4081),
            'KPHL': (39.8729, -75.2407),
            'KMIA': (25.7959, -80.2870),
            'KORD': (41.9786, -87.9048),
            'KDFW': (32.8968, -97.0380),
            'KATL': (33.6407, -84.4277),
            'KDEN': (39.8561, -104.6737),
            'KSFO': (37.6213, -122.3790),
            'KBOS': (42.3656, -71.0096),
            'KSEA': (47.4502, -122.3088),
            'KLAS': (36.0840, -115.1537),
            'KMCO': (28.4312, -81.3081)
        }
        return airport_coords.get(airport_icao)
    
    def _determine_aircraft_phase(self, aircraft: Dict) -> str:
        """Determine aircraft operational phase"""
        altitude = aircraft.get('alt_baro', 0)
        speed = aircraft.get('gs', 0)
        
        if altitude > 3000:
            return 'approach' if speed < 200 else 'departure'
        elif altitude < 100:
            return 'gate' if speed < 10 else 'taxi'
        else:
            return 'taxi'
    
    def _define_airport_zones(self) -> List[Dict]:
        """Define typical airport bottleneck zones"""
        return [
            {
                'zone_id': 'runway_approach_09L',
                'zone_type': 'runway_approach',
                'center_lat': 40.6413,
                'center_lon': -73.7781,
                'radius_meters': 500,
                'capacity': 3
            },
            {
                'zone_id': 'runway_departure_09L',
                'zone_type': 'runway_departure', 
                'center_lat': 40.6413,
                'center_lon': -73.7781,
                'radius_meters': 300,
                'capacity': 2
            },
            {
                'zone_id': 'taxiway_intersection_alpha',
                'zone_type': 'taxiway_intersection',
                'center_lat': 40.6413,
                'center_lon': -73.7781,
                'radius_meters': 200,
                'capacity': 1
            },
            {
                'zone_id': 'gate_area_terminal_1',
                'zone_type': 'gate_area',
                'center_lat': 40.6413,
                'center_lon': -73.7781,
                'radius_meters': 400,
                'capacity': 8
            }
        ]
    
    def _get_aircraft_in_zone(self, flights: List[AircraftData], zone_config: Dict) -> List[AircraftData]:
        """Get aircraft within a specific zone"""
        zone_aircraft = []
        center_lat = zone_config['center_lat']
        center_lon = zone_config['center_lon']
        radius_meters = zone_config['radius_meters']
        
        for flight in flights:
            dist = geopy.distance.distance(
                (center_lat, center_lon),
                (flight.latitude, flight.longitude)
            ).meters
            
            if dist <= radius_meters:
                zone_aircraft.append(flight)
                
        return zone_aircraft
    
    def _calculate_bottleneck_probability(self, zone_aircraft: List[AircraftData], 
                                       zone_config: Dict) -> float:
        """Calculate bottleneck probability for a zone"""
        capacity = zone_config['capacity']
        current_load = len(zone_aircraft)
        
        if current_load == 0:
            return 0.0
            
        # Simple capacity-based probability
        utilization = current_load / capacity
        if utilization >= 1.0:
            return 1.0
        elif utilization >= 0.8:
            return 0.8
        else:
            return utilization * 0.5
    
    def _encode_aircraft_type(self, aircraft_type: str) -> float:
        """Encode aircraft type as numerical value"""
        type_mapping = {
            'B737': 0.1, 'A320': 0.2, 'B777': 0.3, 'A380': 0.4,
            'CRJ9': 0.5, 'E175': 0.6, 'B767F': 0.7
        }
        return type_mapping.get(aircraft_type, 0.0)
    
    def _encode_phase(self, phase: str) -> float:
        """Encode aircraft phase as numerical value"""
        phase_mapping = {
            'approach': 0.1, 'departure': 0.2, 'taxi': 0.3, 
            'gate': 0.4, 'holding': 0.5
        }
        return phase_mapping.get(phase, 0.0)
