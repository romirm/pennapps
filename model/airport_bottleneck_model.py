"""
Complete Airport Bottleneck Analysis Model

Main model that integrates GNN bottleneck detection with KAN impact prediction
to provide comprehensive bottleneck analysis and recommendations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import numpy as np

from .adsb_processor import ADSBDataProcessor, AircraftData
from .bottleneck_gnn import BottleneckGNN
from .kan_predictor import BottleneckKANPredictor
from .flight_metadata import FlightMetadataProcessor
from .impact_calculator import ImpactCalculator


class AirportBottleneckModel(nn.Module):
    """
    Complete bottleneck analysis model integrating GNN and KAN components
    
    Provides:
    - Bottleneck detection and prediction
    - Impact analysis (economic, environmental, operational)
    - Mitigation recommendations
    - Real-time monitoring capabilities
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.config = config
        
        # Initialize components
        self.adsb_processor = ADSBDataProcessor(config)
        self.gnn = BottleneckGNN(
            input_dim=config.get('gnn_input_dim', 5),
            hidden_dim=config.get('gnn_hidden_dim', 64),
            num_layers=config.get('gnn_layers', 4),
            output_dim=config.get('gnn_output_dim', 32)
        )
        self.kan_predictor = BottleneckKANPredictor(
            gnn_embedding_dim=config.get('gnn_output_dim', 32),
            flight_metadata_dim=config.get('flight_metadata_dim', 10),
            output_dim=config.get('kan_output_dim', 64)
        )
        self.flight_metadata_processor = FlightMetadataProcessor()
        self.impact_calculator = ImpactCalculator()
        
        # Model parameters
        self.bottleneck_threshold = config.get('bottleneck_detection_threshold', 0.6)
        self.prediction_horizon = config.get('impact_prediction_horizon_minutes', 60)
        
    def predict_bottlenecks(self, adsb_data: Dict, airport_config: Dict) -> Dict[str, Any]:
        """
        Main prediction function for bottleneck analysis
        
        Args:
            adsb_data: ADS-B data from API
            airport_config: Airport configuration data
            
        Returns:
            Complete bottleneck analysis results
        """
        airport_icao = airport_config.get('icao', 'KJFK')
        
        # 1. Process ADS-B data
        aircraft_list = self.adsb_processor.filter_airport_operations(adsb_data, airport_icao)
        
        if len(aircraft_list) == 0:
            return self._create_empty_analysis(airport_icao)
        
        # 2. Identify bottleneck zones
        bottleneck_zones = self.adsb_processor.identify_bottleneck_zones(aircraft_list)
        
        # 3. Build spatial graph
        graph_data = self.adsb_processor.construct_bottleneck_graph(aircraft_list, airport_config)
        
        # 4. Extract bottleneck embeddings using GNN
        with torch.no_grad():
            bottleneck_predictions = self.gnn(graph_data)
        
        # 5. Process flight metadata
        flight_metadata = self._process_flight_metadata(aircraft_list)
        
        # 6. Predict bottleneck impacts using KAN
        impact_predictions = self.kan_predictor(
            bottleneck_predictions['bottleneck_embeddings'],
            flight_metadata
        )
        
        # 7. Calculate comprehensive impacts
        comprehensive_analysis = self._calculate_comprehensive_impacts(
            aircraft_list, bottleneck_zones, impact_predictions
        )
        
        # 8. Generate recommendations
        recommendations = self._generate_recommendations(
            bottleneck_zones, impact_predictions, comprehensive_analysis
        )
        
        # 9. Format final output
        return self._format_analysis_output(
            airport_icao, aircraft_list, bottleneck_zones, 
            impact_predictions, comprehensive_analysis, recommendations
        )
    
    def _process_flight_metadata(self, aircraft_list: List[AircraftData]) -> torch.Tensor:
        """Process flight metadata into tensor format"""
        metadata_list = []
        
        for aircraft in aircraft_list:
            flight_data = {
                'aircraft_type': aircraft.aircraft_type,
                'phase': aircraft.phase,
                'altitude': aircraft.altitude,
                'speed': aircraft.speed,
                'heading': aircraft.heading
            }
            
            # Process metadata
            processed_metadata = self.flight_metadata_processor.process_flight_metadata(flight_data)
            
            # Convert to tensor features
            features = [
                processed_metadata.get('estimated_passengers', 0) / 1000.0,  # Normalize
                processed_metadata.get('fuel_capacity_gallons', 0) / 10000.0,  # Normalize
                processed_metadata.get('fuel_burn_idle_gph', 0) / 1000.0,  # Normalize
                processed_metadata.get('load_factor', 0),
                aircraft.altitude / 1000.0,  # Normalize altitude
                aircraft.speed / 100.0,  # Normalize speed
                aircraft.heading / 360.0,  # Normalize heading
                self._encode_phase(aircraft.phase),
                self._encode_aircraft_category(processed_metadata.get('aircraft_category', 'unknown')),
                processed_metadata.get('max_range_nm', 0) / 10000.0  # Normalize range
            ]
            
            metadata_list.append(features)
        
        if len(metadata_list) == 0:
            return torch.zeros((1, 10))
        
        return torch.tensor(metadata_list, dtype=torch.float32)
    
    def _calculate_comprehensive_impacts(self, aircraft_list: List[AircraftData], 
                                       bottleneck_zones: List, 
                                       impact_predictions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Calculate comprehensive impact analysis"""
        comprehensive_analysis = {
            'economic_impact': {},
            'environmental_impact': {},
            'operational_impact': {},
            'passenger_impact': {}
        }
        
        # Convert aircraft data to flight dictionaries for impact calculation
        flight_data_list = []
        for aircraft in aircraft_list:
            flight_data = {
                'aircraft_type': aircraft.aircraft_type,
                'phase': aircraft.phase,
                'estimated_passengers': self.flight_metadata_processor.estimate_passenger_count({
                    'aircraft_type': aircraft.aircraft_type,
                    'phase': aircraft.phase
                })
            }
            flight_data_list.append(flight_data)
        
        # Calculate impacts for each bottleneck zone
        for zone in bottleneck_zones:
            bottleneck_data = {
                'type': zone.zone_type,
                'severity': zone.bottleneck_probability,
                'delay_minutes': self._estimate_delay_minutes(zone),
                'zone_id': zone.zone_id
            }
            
            # Economic impact
            economic_impact = self.impact_calculator.calculate_economic_impact(
                bottleneck_data, flight_data_list
            )
            comprehensive_analysis['economic_impact'][zone.zone_id] = economic_impact
            
            # Environmental impact
            environmental_impact = self.impact_calculator.calculate_environmental_impact(
                bottleneck_data, flight_data_list
            )
            comprehensive_analysis['environmental_impact'][zone.zone_id] = environmental_impact
            
            # Operational impact
            operational_impact = self.impact_calculator.calculate_operational_impact(
                bottleneck_data, {}
            )
            comprehensive_analysis['operational_impact'][zone.zone_id] = operational_impact
            
            # Passenger impact
            passenger_impact = self.impact_calculator.calculate_passenger_impact(
                flight_data_list, bottleneck_data.get('delay_minutes', 0)
            )
            comprehensive_analysis['passenger_impact'][zone.zone_id] = passenger_impact
        
        return comprehensive_analysis
    
    def _generate_recommendations(self, bottleneck_zones: List, 
                                impact_predictions: Dict[str, torch.Tensor],
                                comprehensive_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate mitigation recommendations"""
        recommendations = []
        
        for zone in bottleneck_zones:
            bottleneck_data = {
                'type': zone.zone_type,
                'severity': zone.bottleneck_probability,
                'zone_id': zone.zone_id
            }
            
            # Generate mitigation options
            mitigation_options = self.kan_predictor.generate_mitigation_recommendations(bottleneck_data)
            
            # Calculate cost-benefit analysis
            cost_benefit_analysis = self.impact_calculator.calculate_mitigation_cost_benefit(
                bottleneck_data, mitigation_options
            )
            
            recommendations.append({
                'zone_id': zone.zone_id,
                'bottleneck_type': zone.zone_type,
                'severity': zone.bottleneck_probability,
                'mitigation_options': mitigation_options,
                'cost_benefit_analysis': cost_benefit_analysis,
                'recommended_action': self._select_best_mitigation(cost_benefit_analysis)
            })
        
        return recommendations
    
    def _format_analysis_output(self, airport_icao: str, aircraft_list: List[AircraftData],
                              bottleneck_zones: List, impact_predictions: Dict[str, torch.Tensor],
                              comprehensive_analysis: Dict[str, Any], 
                              recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Format final analysis output"""
        
        # Calculate summary metrics
        total_aircraft = len(aircraft_list)
        total_bottlenecks = len(bottleneck_zones)
        highest_severity = max([zone.bottleneck_probability for zone in bottleneck_zones]) if bottleneck_zones else 0
        
        # Calculate total passengers at risk
        total_passengers = sum([
            self.flight_metadata_processor.estimate_passenger_count({
                'aircraft_type': aircraft.aircraft_type,
                'phase': aircraft.phase
            }) for aircraft in aircraft_list
        ])
        
        # Calculate total fuel waste estimate
        total_fuel_waste = 0
        for zone_id, economic_impact in comprehensive_analysis['economic_impact'].items():
            total_fuel_waste += economic_impact.get('direct_costs', {}).get('fuel_waste_cost', 0) / 3.50  # Convert cost to gallons
        
        # Determine overall delay risk
        if highest_severity > 0.8:
            overall_delay_risk = 'critical'
        elif highest_severity > 0.6:
            overall_delay_risk = 'high'
        elif highest_severity > 0.4:
            overall_delay_risk = 'medium'
        else:
            overall_delay_risk = 'low'
        
        # Format bottleneck predictions
        bottleneck_predictions = []
        for i, zone in enumerate(bottleneck_zones):
            # Get impact predictions for this zone
            zone_impacts = {
                'economic': comprehensive_analysis['economic_impact'].get(zone.zone_id, {}),
                'environmental': comprehensive_analysis['environmental_impact'].get(zone.zone_id, {}),
                'operational': comprehensive_analysis['operational_impact'].get(zone.zone_id, {}),
                'passenger': comprehensive_analysis['passenger_impact'].get(zone.zone_id, {})
            }
            
            # Get aircraft affected in this zone
            zone_aircraft = self.adsb_processor._get_aircraft_in_zone(aircraft_list, {
                'center_lat': zone.center_lat,
                'center_lon': zone.center_lon,
                'radius_meters': zone.radius_meters
            })
            
            aircraft_affected = []
            for aircraft in zone_aircraft:
                aircraft_affected.append({
                    'flight_id': aircraft.flight_id,
                    'aircraft_type': aircraft.aircraft_type,
                    'estimated_passengers': self.flight_metadata_processor.estimate_passenger_count({
                        'aircraft_type': aircraft.aircraft_type,
                        'phase': aircraft.phase
                    }),
                    'delay_contribution': zone.bottleneck_probability * 10,  # Simplified
                    'current_phase': aircraft.phase
                })
            
            # Get recommendations for this zone
            zone_recommendations = next(
                (rec for rec in recommendations if rec['zone_id'] == zone.zone_id), 
                {'mitigation_options': [], 'recommended_action': None}
            )
            
            bottleneck_prediction = {
                'bottleneck_id': zone.zone_id,
                'location': {
                    'zone': zone.zone_type,
                    'coordinates': [zone.center_lat, zone.center_lon]
                },
                'type': zone.zone_type,
                'probability': float(zone.bottleneck_probability),
                'severity': int(zone.bottleneck_probability * 5) + 1,  # Convert to 1-5 scale
                
                'timing': {
                    'predicted_onset_minutes': 0,  # Already occurring
                    'estimated_duration_minutes': self._estimate_delay_minutes(zone),
                    'resolution_confidence': 1.0 - zone.bottleneck_probability
                },
                
                'aircraft_affected': aircraft_affected,
                
                'impact_analysis': {
                    'passengers_affected': zone_impacts['passenger'].get('passengers_affected', 0),
                    'total_delay_minutes': self._estimate_delay_minutes(zone),
                    'fuel_waste_gallons': zone_impacts['environmental'].get('fuel_waste_gallons', 0),
                    'fuel_cost_estimate': zone_impacts['economic'].get('direct_costs', {}).get('fuel_waste_cost', 0),
                    'co2_emissions_lbs': zone_impacts['environmental'].get('co2_emissions_lbs', 0),
                    'economic_impact_estimate': zone_impacts['economic'].get('total_economic_impact_usd', 0)
                },
                
                'recommended_mitigations': zone_recommendations['mitigation_options']
            }
            
            bottleneck_predictions.append(bottleneck_prediction)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'airport': airport_icao,
            'analysis_radius_nm': self.config.get('adsb_radius_nm', 3),
            'total_aircraft_monitored': total_aircraft,
            
            'bottleneck_predictions': bottleneck_predictions,
            
            'airport_summary': {
                'total_bottlenecks_predicted': total_bottlenecks,
                'highest_severity_level': int(highest_severity * 5) + 1,
                'total_passengers_at_risk': total_passengers,
                'total_fuel_waste_estimate': total_fuel_waste,
                'overall_delay_risk': overall_delay_risk
            },
            
            'recommendations_summary': recommendations
        }
    
    def _create_empty_analysis(self, airport_icao: str) -> Dict[str, Any]:
        """Create empty analysis when no aircraft are detected"""
        return {
            'timestamp': datetime.now().isoformat(),
            'airport': airport_icao,
            'analysis_radius_nm': self.config.get('adsb_radius_nm', 3),
            'total_aircraft_monitored': 0,
            'bottleneck_predictions': [],
            'airport_summary': {
                'total_bottlenecks_predicted': 0,
                'highest_severity_level': 1,
                'total_passengers_at_risk': 0,
                'total_fuel_waste_estimate': 0,
                'overall_delay_risk': 'low'
            },
            'recommendations_summary': []
        }
    
    def _estimate_delay_minutes(self, zone) -> float:
        """Estimate delay minutes for a bottleneck zone"""
        base_delays = {
            'runway_approach_queue': 15,
            'runway_departure_queue': 10,
            'taxiway_intersection': 5,
            'gate_availability': 20
        }
        
        base_delay = base_delays.get(zone.zone_type, 10)
        return base_delay * zone.bottleneck_probability
    
    def _encode_phase(self, phase: str) -> float:
        """Encode aircraft phase as numerical value"""
        phase_mapping = {
            'approach': 0.1, 'departure': 0.2, 'taxi': 0.3, 
            'gate': 0.4, 'holding': 0.5
        }
        return phase_mapping.get(phase, 0.0)
    
    def _encode_aircraft_category(self, category: str) -> float:
        """Encode aircraft category as numerical value"""
        category_mapping = {
            'narrow_body': 0.1, 'wide_body': 0.2, 'regional': 0.3, 'cargo': 0.4
        }
        return category_mapping.get(category, 0.0)
    
    def _select_best_mitigation(self, cost_benefit_analysis: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Select the best mitigation option based on cost-benefit analysis"""
        if not cost_benefit_analysis:
            return None
        
        # Sort by ROI (return on investment)
        sorted_options = sorted(cost_benefit_analysis, key=lambda x: x['roi_percentage'], reverse=True)
        
        # Return the highest ROI option with reasonable implementation time
        for option in sorted_options:
            if option['implementation_time_minutes'] < 30:  # Less than 30 minutes to implement
                return option
        
        # If no quick options, return the highest ROI
        return sorted_options[0] if sorted_options else None
    
    def save_model(self, filepath: str):
        """Save the complete model"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.config
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load the complete model"""
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.config.update(checkpoint['config'])
