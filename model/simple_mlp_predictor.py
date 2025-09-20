"""
Simple MLP Bottleneck Predictor

A straightforward Multi-Layer Perceptron that replaces the complex GNN-KAN hybrid
with a simple neural network for bottleneck prediction and impact analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class BottleneckPrediction:
    """Simple data class for bottleneck predictions"""
    bottleneck_id: str
    zone_type: str
    probability: float
    severity: int  # 1-5 scale
    estimated_delay_minutes: float
    aircraft_count: int
    passengers_affected: int
    fuel_waste_gallons: float
    coordinates: Tuple[float, float]


class SimpleMLPPredictor(nn.Module):
    """
    Simple Multi-Layer Perceptron for bottleneck prediction
    
    Replaces the complex GNN-KAN architecture with a straightforward
    neural network that takes aircraft features and predicts bottlenecks.
    """
    
    def __init__(self, input_dim: int = 20, hidden_dim: int = 128, output_dim: int = 8):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Main MLP architecture
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layers
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output layer
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)


class SimpleBottleneckPredictor:
    """
    Main bottleneck prediction class using simple MLP
    
    This replaces the complex GNN-KAN hybrid with a straightforward
    approach that's easier to understand, debug, and maintain.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize MLP model
        self.mlp = SimpleMLPPredictor(
            input_dim=config.get('mlp_input_dim', 20),
            hidden_dim=config.get('mlp_hidden_dim', 128),
            output_dim=config.get('mlp_output_dim', 8)
        )
        
        # Bottleneck zone definitions (simplified)
        self.zones = {
            'runway_approach': {'center': [0, 0], 'radius': 2.0},
            'taxiway_intersection': {'center': [0, 0], 'radius': 0.5},
            'gate_area': {'center': [0, 0], 'radius': 1.0},
            'departure_queue': {'center': [0, 0], 'radius': 1.5}
        }
        
        # Thresholds for bottleneck detection
        self.thresholds = {
            'density_threshold': 0.6,
            'delay_threshold_minutes': 5,
            'fuel_waste_threshold_gallons': 50
        }
    
    def extract_features(self, aircraft_data: List[Dict], airport_coords: Tuple[float, float]) -> np.ndarray:
        """
        Extract features from aircraft data for MLP input
        
        Args:
            aircraft_data: List of aircraft dictionaries from ADS-B
            airport_coords: (lat, lon) of airport center
            
        Returns:
            Feature vector for MLP input
        """
        if not aircraft_data:
            return np.zeros(20)  # Return zero vector if no aircraft
        
        features = []
        
        # Basic aircraft count and statistics
        aircraft_count = len(aircraft_data)
        features.append(aircraft_count)
        
        # Speed statistics
        speeds = [ac.get('gs', 0) for ac in aircraft_data]
        features.extend([
            np.mean(speeds) if speeds else 0,
            np.std(speeds) if len(speeds) > 1 else 0,
            np.max(speeds) if speeds else 0
        ])
        
        # Altitude statistics
        altitudes = [ac.get('alt_baro', 0) for ac in aircraft_data]
        features.extend([
            np.mean(altitudes) if altitudes else 0,
            np.std(altitudes) if len(altitudes) > 1 else 0
        ])
        
        # Position spread (relative to airport)
        lats = [ac.get('lat', airport_coords[0]) for ac in aircraft_data]
        lons = [ac.get('lon', airport_coords[1]) for ac in aircraft_data]
        
        # Calculate distances from airport center
        distances = []
        for lat, lon in zip(lats, lons):
            dist = self._haversine_distance(lat, lon, airport_coords[0], airport_coords[1])
            distances.append(dist)
        
        features.extend([
            np.mean(distances) if distances else 0,
            np.std(distances) if len(distances) > 1 else 0,
            np.max(distances) if distances else 0
        ])
        
        # Aircraft type distribution (simplified)
        aircraft_types = [ac.get('t', 'UNKNOWN') for ac in aircraft_data]
        type_counts = {
            'B737': aircraft_types.count('B737'),
            'A320': aircraft_types.count('A320'),
            'B777': aircraft_types.count('B777'),
            'OTHER': len(aircraft_types) - sum([aircraft_types.count(t) for t in ['B737', 'A320', 'B777']])
        }
        
        features.extend([
            type_counts['B737'] / aircraft_count if aircraft_count > 0 else 0,
            type_counts['A320'] / aircraft_count if aircraft_count > 0 else 0,
            type_counts['B777'] / aircraft_count if aircraft_count > 0 else 0,
            type_counts['OTHER'] / aircraft_count if aircraft_count > 0 else 0
        ])
        
        # Track/heading statistics
        tracks = [ac.get('track', 0) for ac in aircraft_data if ac.get('track') is not None]
        features.extend([
            np.mean(tracks) if tracks else 0,
            np.std(tracks) if len(tracks) > 1 else 0
        ])
        
        # Time-based features (if available)
        features.extend([
            0,  # Hour of day (placeholder)
            0,  # Day of week (placeholder)
            0   # Season (placeholder)
        ])
        
        # Ensure we have exactly 20 features
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20])
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Earth radius in nautical miles
        r = 3440.065
        return c * r
    
    def predict_bottlenecks(self, aircraft_data: List[Dict], airport_coords: Tuple[float, float]) -> List[BottleneckPrediction]:
        """
        Predict bottlenecks using simple MLP
        
        Args:
            aircraft_data: List of aircraft from ADS-B
            airport_coords: Airport center coordinates
            
        Returns:
            List of bottleneck predictions
        """
        # Extract features
        features = self.extract_features(aircraft_data, airport_coords)
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Get MLP predictions
        with torch.no_grad():
            predictions = self.mlp(features_tensor)
            predictions = torch.sigmoid(predictions)  # Convert to probabilities
        
        # Parse predictions
        bottleneck_predictions = []
        
        # Output format: [runway_prob, taxiway_prob, gate_prob, departure_prob, 
        #                  delay_minutes, passengers, fuel_waste, severity]
        pred_values = predictions[0].numpy()
        
        zone_types = ['runway_approach', 'taxiway_intersection', 'gate_area', 'departure_queue']
        
        for i, zone_type in enumerate(zone_types):
            probability = float(pred_values[i])
            
            if probability > self.thresholds['density_threshold']:
                # Calculate impact metrics
                aircraft_count = len(aircraft_data)
                estimated_delay = float(pred_values[4]) * 15  # Scale to minutes
                passengers_affected = int(pred_values[5] * aircraft_count * 150)  # Assume 150 passengers per aircraft
                fuel_waste = float(pred_values[6]) * 100  # Scale to gallons
                severity = int(pred_values[7] * 5) + 1  # Scale to 1-5
                
                prediction = BottleneckPrediction(
                    bottleneck_id=f"{zone_type}_{aircraft_count}",
                    zone_type=zone_type,
                    probability=probability,
                    severity=severity,
                    estimated_delay_minutes=estimated_delay,
                    aircraft_count=aircraft_count,
                    passengers_affected=passengers_affected,
                    fuel_waste_gallons=fuel_waste,
                    coordinates=airport_coords
                )
                
                bottleneck_predictions.append(prediction)
        
        return bottleneck_predictions
    
    def get_airport_summary(self, predictions: List[BottleneckPrediction]) -> Dict[str, Any]:
        """Generate airport-level summary from predictions"""
        if not predictions:
            return {
                'total_bottlenecks_predicted': 0,
                'highest_severity_level': 0,
                'total_passengers_at_risk': 0,
                'total_fuel_waste_estimate': 0,
                'overall_delay_risk': 'low'
            }
        
        total_bottlenecks = len(predictions)
        highest_severity = max(pred.severity for pred in predictions)
        total_passengers = sum(pred.passengers_affected for pred in predictions)
        total_fuel_waste = sum(pred.fuel_waste_gallons for pred in predictions)
        
        # Determine overall risk
        if highest_severity >= 4:
            overall_risk = 'critical'
        elif highest_severity >= 3:
            overall_risk = 'high'
        elif highest_severity >= 2:
            overall_risk = 'medium'
        else:
            overall_risk = 'low'
        
        return {
            'total_bottlenecks_predicted': total_bottlenecks,
            'highest_severity_level': highest_severity,
            'total_passengers_at_risk': total_passengers,
            'total_fuel_waste_estimate': total_fuel_waste,
            'overall_delay_risk': overall_risk
        }
    
    def save_model(self, filepath: str):
        """Save the MLP model"""
        torch.save({
            'model_state_dict': self.mlp.state_dict(),
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the MLP model"""
        checkpoint = torch.load(filepath)
        self.mlp.load_state_dict(checkpoint['model_state_dict'])
        self.config.update(checkpoint['config'])
