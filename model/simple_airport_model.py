"""
Simplified Airport Bottleneck Analysis Model

Replaces the complex GNN-KAN hybrid with a simple MLP-based approach
that's easier to understand, debug, and maintain.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import numpy as np

from simple_mlp_predictor import SimpleBottleneckPredictor, BottleneckPrediction
try:
    from flight_metadata import FlightMetadataProcessor
    from impact_calculator import ImpactCalculator
except ImportError:
    # Create dummy classes if modules not available
    class FlightMetadataProcessor:
        def __init__(self, config=None): pass
        def get_aircraft_metadata(self, aircraft): return {}
    
    class ImpactCalculator:
        def __init__(self, config=None): pass


class SimpleAirportBottleneckModel:
    """
    Simplified bottleneck analysis model using MLP instead of GNN-KAN hybrid
    
    Provides:
    - Simple bottleneck detection and prediction
    - Impact analysis (economic, environmental, operational)
    - Mitigation recommendations
    - Real-time monitoring capabilities
    """
    
    def __init__(self, config: Dict):
        self.config = config
        
        # Initialize simplified components
        self.bottleneck_predictor = SimpleBottleneckPredictor(config)
        self.flight_metadata = FlightMetadataProcessor()
        self.impact_calculator = ImpactCalculator()
        
        # Model state
        self.is_trained = False
    
    def process_adsb_data(self, adsb_data: Dict, airport_code: str) -> Dict[str, Any]:
        """
        Process ADS-B data and extract aircraft information
        
        Args:
            adsb_data: Raw ADS-B data from API
            airport_code: ICAO airport code
            
        Returns:
            Processed aircraft data
        """
        aircraft_list = []
        
        if 'aircraft' in adsb_data:
            for aircraft in adsb_data['aircraft']:
                # Basic aircraft data extraction
                aircraft_info = {
                    'flight': aircraft.get('flight', 'UNKNOWN'),
                    'aircraft_type': aircraft.get('t', 'UNKNOWN'),
                    'lat': aircraft.get('lat', 0),
                    'lon': aircraft.get('lon', 0),
                    'alt_baro': aircraft.get('alt_baro', 0),
                    'track': aircraft.get('track', 0),
                    'gs': aircraft.get('gs', 0),
                    'timestamp': aircraft.get('timestamp', datetime.now().isoformat())
                }
                
                # Add metadata
                metadata = self.flight_metadata.process_flight_metadata(aircraft_info)
                aircraft_info.update(metadata)
                
                aircraft_list.append(aircraft_info)
        
        return {
            'aircraft': aircraft_list,
            'airport_code': airport_code,
            'total_aircraft': len(aircraft_list),
            'timestamp': datetime.now().isoformat()
        }
    
    def predict_bottlenecks(self, adsb_data: Dict, airport_config: Dict) -> Dict[str, Any]:
        """
        Main prediction function - simplified version of the complex GNN-KAN approach
        
        Args:
            adsb_data: ADS-B data from API
            airport_config: Airport configuration
            
        Returns:
            Comprehensive bottleneck analysis
        """
        airport_code = airport_config.get('icao', 'UNKNOWN')
        
        # Process ADS-B data
        processed_data = self.process_adsb_data(adsb_data, airport_code)
        aircraft_list = processed_data['aircraft']
        
        # Get airport coordinates (simplified)
        airport_coords = self._get_airport_coordinates(airport_code)
        
        # Predict bottlenecks using simple MLP
        bottleneck_predictions = self.bottleneck_predictor.predict_bottlenecks(
            aircraft_list, airport_coords
        )
        
        # Generate airport summary
        airport_summary = self.bottleneck_predictor.get_airport_summary(bottleneck_predictions)
        
        # Format predictions for output
        formatted_predictions = []
        for pred in bottleneck_predictions:
            formatted_pred = {
                'bottleneck_id': pred.bottleneck_id,
                'location': {
                    'zone': pred.zone_type,
                    'coordinates': pred.coordinates
                },
                'type': pred.zone_type,
                'probability': pred.probability,
                'severity': pred.severity,
                
                'timing': {
                    'predicted_onset_minutes': 0,
                    'estimated_duration_minutes': pred.estimated_delay_minutes,
                    'resolution_confidence': pred.probability
                },
                
                'aircraft_affected': [
                    {
                        'flight_id': aircraft.get('flight', 'UNKNOWN'),
                        'aircraft_type': aircraft.get('aircraft_type', 'UNKNOWN'),
                        'estimated_passengers': aircraft.get('estimated_passengers', 150),
                        'delay_contribution': pred.estimated_delay_minutes / len(aircraft_list) if aircraft_list else 0,
                        'current_phase': self._determine_aircraft_phase(aircraft)
                    }
                    for aircraft in aircraft_list[:5]  # Limit to first 5 aircraft
                ],
                
                'impact_analysis': {
                    'passengers_affected': pred.passengers_affected,
                    'total_delay_minutes': pred.estimated_delay_minutes,
                    'fuel_waste_gallons': pred.fuel_waste_gallons,
                    'fuel_cost_estimate': pred.fuel_waste_gallons * self.config.get('fuel_price_per_gallon', 3.50),
                    'co2_emissions_lbs': pred.fuel_waste_gallons * self.config.get('co2_per_gallon_fuel', 21.1),
                    'economic_impact_estimate': self._calculate_economic_impact(pred)
                },
                
                'recommended_mitigations': self._generate_mitigations(pred)
            }
            formatted_predictions.append(formatted_pred)
        
        # Return comprehensive analysis
        return {
            'timestamp': datetime.now().isoformat(),
            'airport': airport_code,
            'analysis_radius_nm': self.config.get('adsb_radius_nm', 3),
            'total_aircraft_monitored': len(aircraft_list),
            
            'bottleneck_predictions': formatted_predictions,
            'airport_summary': airport_summary,
            
            'model_info': {
                'model_type': 'Simple MLP',
                'version': '1.0',
                'is_trained': self.is_trained
            }
        }
    
    def _get_airport_coordinates(self, airport_code: str) -> Tuple[float, float]:
        """Get airport coordinates (simplified)"""
        # Simple airport coordinate lookup
        airport_coords = {
            'KJFK': (40.63980103, -73.77890015),
            'KLAX': (33.9425, -118.4081),
            'KORD': (41.9786, -87.9048),
            'KDFW': (32.8968, -97.0380),
            'KPHL': (39.8719, -75.2411),
            'KATL': (33.6407, -84.4277),
            'KLGA': (40.7769, -73.8740),
            'KBOS': (42.3656, -71.0096)
        }
        
        return airport_coords.get(airport_code, (0.0, 0.0))
    
    def _determine_aircraft_phase(self, aircraft: Dict) -> str:
        """Determine aircraft operational phase"""
        altitude = aircraft.get('alt_baro', 0)
        speed = aircraft.get('gs', 0)
        
        if altitude > 3000:
            return 'cruise'
        elif altitude > 1000:
            return 'approach'
        elif speed < 50:
            return 'taxi'
        else:
            return 'departure'
    
    def _calculate_economic_impact(self, prediction: BottleneckPrediction) -> float:
        """Calculate economic impact of bottleneck"""
        fuel_cost = prediction.fuel_waste_gallons * self.config.get('fuel_price_per_gallon', 3.50)
        passenger_cost = prediction.passengers_affected * prediction.estimated_delay_minutes * self.config.get('passenger_compensation_rate', 2.50)
        
        return fuel_cost + passenger_cost
    
    def _generate_mitigations(self, prediction: BottleneckPrediction) -> List[Dict]:
        """Generate mitigation recommendations"""
        mitigations = []
        
        if prediction.zone_type == 'runway_approach':
            mitigations.append({
                'action': 'Increase runway separation',
                'priority': 'high',
                'estimated_effectiveness': 0.8,
                'implementation_time': 5.0
            })
        elif prediction.zone_type == 'taxiway_intersection':
            mitigations.append({
                'action': 'Optimize taxiway sequencing',
                'priority': 'medium',
                'estimated_effectiveness': 0.6,
                'implementation_time': 3.0
            })
        elif prediction.zone_type == 'gate_area':
            mitigations.append({
                'action': 'Reassign gate operations',
                'priority': 'medium',
                'estimated_effectiveness': 0.7,
                'implementation_time': 10.0
            })
        
        return mitigations
    
    def train_model(self, training_data: List[Dict]):
        """
        Train the MLP model (placeholder for future implementation)
        
        Args:
            training_data: Historical bottleneck data for training
        """
        # TODO: Implement training logic
        print("Training functionality not yet implemented")
        self.is_trained = True
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        self.bottleneck_predictor.save_model(filepath)
    
    def load_model(self, filepath: str):
        """Load the complete model"""
        self.bottleneck_predictor.load_model(filepath)
        self.is_trained = True


# Backward compatibility - create alias for existing code
AirportBottleneckModel = SimpleAirportBottleneckModel
