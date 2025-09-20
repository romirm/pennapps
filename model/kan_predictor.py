"""
KAN Impact Predictor

Kolmogorov-Arnold Network (KAN) model for predicting bottleneck characteristics
and impact metrics including resolution time, passenger impact, and fuel waste.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class KANLayer(nn.Module):
    """
    Simplified KAN layer implementation
    In production, use proper KAN implementation or MLP approximation
    """
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # MLP approximation of KAN functionality
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable basis functions (simplified)
        self.basis_functions = nn.Parameter(torch.randn(input_dim, 8))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply basis functions
        basis_output = torch.matmul(x, self.basis_functions)
        basis_output = torch.sin(basis_output)  # Sinusoidal basis
        
        # Combine with MLP
        mlp_output = self.mlp(x)
        
        # Weighted combination
        combined = 0.7 * mlp_output + 0.3 * basis_output.mean(dim=-1, keepdim=True)
        
        return combined


class BottleneckKANPredictor(nn.Module):
    """
    KAN-based predictor for bottleneck impact analysis
    
    Predicts:
    - Bottleneck probability (0-1)
    - Resolution time (minutes)
    - Passengers affected
    - Fuel waste estimates
    - Severity level (1-5)
    - Bottleneck type (categorical)
    """
    
    def __init__(self, gnn_embedding_dim: int = 32, flight_metadata_dim: int = 10, 
                 output_dim: int = 64):
        super().__init__()
        
        self.gnn_embedding_dim = gnn_embedding_dim
        self.flight_metadata_dim = flight_metadata_dim
        self.output_dim = output_dim
        
        # Input processing layers
        self.gnn_processor = KANLayer(gnn_embedding_dim, output_dim // 2)
        self.metadata_processor = KANLayer(flight_metadata_dim, output_dim // 2)
        
        # Impact prediction heads
        self.impact_heads = nn.ModuleDict({
            'bottleneck_probability': nn.Sequential(
                KANLayer(output_dim, 32),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'resolution_time_minutes': nn.Sequential(
                KANLayer(output_dim, 32),
                nn.Linear(32, 1),
                nn.ReLU()  # Non-negative time
            ),
            'passengers_affected': nn.Sequential(
                KANLayer(output_dim, 32),
                nn.Linear(32, 1),
                nn.ReLU()  # Non-negative count
            ),
            'fuel_waste_gallons': nn.Sequential(
                KANLayer(output_dim, 32),
                nn.Linear(32, 1),
                nn.ReLU()  # Non-negative fuel
            ),
            'severity_level': nn.Sequential(
                KANLayer(output_dim, 32),
                nn.Linear(32, 5),
                nn.Softmax(dim=-1)
            ),
            'bottleneck_type': nn.Sequential(
                KANLayer(output_dim, 32),
                nn.Linear(32, 4),  # 4 bottleneck types
                nn.Softmax(dim=-1)
            )
        })
        
        # Economic impact calculator
        self.economic_impact = nn.Sequential(
            KANLayer(output_dim, 32),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
        # Environmental impact calculator
        self.environmental_impact = nn.Sequential(
            KANLayer(output_dim, 32),
            nn.Linear(32, 1),
            nn.ReLU()
        )
        
    def forward(self, bottleneck_embeddings: torch.Tensor, 
                flight_metadata: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict bottleneck impacts
        
        Args:
            bottleneck_embeddings: GNN bottleneck embeddings
            flight_metadata: Flight metadata tensor
            
        Returns:
            Dictionary of impact predictions
        """
        # Process inputs
        gnn_features = self.gnn_processor(bottleneck_embeddings)
        metadata_features = self.metadata_processor(flight_metadata)
        
        # Combine features
        combined_features = torch.cat([gnn_features, metadata_features], dim=-1)
        
        # Generate predictions
        predictions = {}
        for head_name, head_network in self.impact_heads.items():
            predictions[head_name] = head_network(combined_features)
        
        # Additional impact calculations
        predictions['economic_impact'] = self.economic_impact(combined_features)
        predictions['environmental_impact'] = self.environmental_impact(combined_features)
        
        return predictions
    
    def calculate_resolution_time(self, bottleneck_type: str, severity: float, 
                                weather_conditions: Dict) -> float:
        """
        Calculate estimated resolution time based on bottleneck characteristics
        
        Args:
            bottleneck_type: Type of bottleneck
            severity: Severity level (0-1)
            weather_conditions: Weather conditions dict
            
        Returns:
            Estimated resolution time in minutes
        """
        base_resolution_times = {
            'runway_approach_queue': 5.0,    # minutes per additional aircraft
            'runway_departure_queue': 3.0,   # minutes per additional aircraft  
            'taxiway_intersection': 2.0,     # minutes per conflict
            'gate_availability': 15.0,        # minutes average gate wait
            'weather_delay': 30.0            # minutes for weather clearing
        }
        
        base_time = base_resolution_times.get(bottleneck_type, 10.0)
        
        # Apply severity multiplier
        resolution_time = base_time * (1.0 + severity * 2.0)
        
        # Apply weather multiplier
        if weather_conditions.get('visibility', 10) < 3:  # miles
            resolution_time *= 1.5
        if weather_conditions.get('wind_speed', 0) > 20:  # knots
            resolution_time *= 1.2
            
        return resolution_time
    
    def calculate_passenger_impact(self, affected_flights: List[Dict], 
                                 delay_minutes: float) -> Dict[str, float]:
        """
        Calculate passenger impact metrics
        
        Args:
            affected_flights: List of affected flight data
            delay_minutes: Total delay in minutes
            
        Returns:
            Dictionary of passenger impact metrics
        """
        total_passengers = sum([
            flight.get('estimated_passengers', 0) 
            for flight in affected_flights
        ])
        
        impact_metrics = {
            'passengers_delayed': total_passengers,
            'missed_connections_estimate': total_passengers * 0.15 * (delay_minutes / 60),
            'compensation_cost_estimate': total_passengers * delay_minutes * 2.50,  # $2.50 per passenger per minute
            'customer_satisfaction_impact': min(delay_minutes / 60 * 0.2, 1.0)  # 0-1 scale
        }
        
        return impact_metrics
    
    def calculate_fuel_waste(self, affected_flights: List[Dict], 
                           delay_minutes: float) -> Dict[str, float]:
        """
        Calculate fuel waste and environmental impact
        
        Args:
            affected_flights: List of affected flight data
            delay_minutes: Total delay in minutes
            
        Returns:
            Dictionary of fuel waste metrics
        """
        aircraft_database = {
            'B737': {'fuel_burn_idle': 150},  # gallons/hour
            'A320': {'fuel_burn_idle': 140},
            'B777': {'fuel_burn_idle': 300},
            'A380': {'fuel_burn_idle': 500},
            'CRJ9': {'fuel_burn_idle': 80},
            'E175': {'fuel_burn_idle': 75},
            'B767F': {'fuel_burn_idle': 250}
        }
        
        total_fuel_waste = 0
        
        for flight in affected_flights:
            aircraft_type = flight.get('aircraft_type', 'B737')
            fuel_burn_rate = aircraft_database.get(aircraft_type, {'fuel_burn_idle': 150})['fuel_burn_idle']
            
            phase = flight.get('phase', 'ground')
            if phase == 'ground':
                fuel_waste = (fuel_burn_rate / 60) * delay_minutes  # gallons
            elif phase == 'holding_pattern':
                fuel_waste = (fuel_burn_rate * 1.5 / 60) * delay_minutes  # 1.5x for airborne holding
            else:
                fuel_waste = (fuel_burn_rate / 60) * delay_minutes
            
            total_fuel_waste += fuel_waste
        
        # Convert to cost and environmental impact
        fuel_cost = total_fuel_waste * 3.50  # $3.50 per gallon average
        co2_emissions = total_fuel_waste * 21.1  # lbs CO2 per gallon
        
        return {
            'fuel_gallons': total_fuel_waste,
            'cost_estimate': fuel_cost,
            'co2_emissions_lbs': co2_emissions
        }
    
    def predict_bottleneck_cascade(self, initial_bottleneck: Dict, 
                                 airport_config: Dict) -> List[Dict]:
        """
        Predict cascading effects of bottlenecks
        
        Args:
            initial_bottleneck: Initial bottleneck data
            airport_config: Airport configuration
            
        Returns:
            List of predicted cascading bottlenecks
        """
        cascade_bottlenecks = []
        
        # Simple cascade prediction (in production, use more sophisticated modeling)
        bottleneck_type = initial_bottleneck.get('type', 'unknown')
        severity = initial_bottleneck.get('severity', 0.5)
        
        if bottleneck_type == 'runway_approach_queue':
            # Runway delays can cascade to gate delays
            cascade_bottlenecks.append({
                'type': 'gate_availability',
                'probability': severity * 0.8,
                'delay_minutes': initial_bottleneck.get('delay_minutes', 0) * 0.5,
                'cascade_source': 'runway_delay'
            })
        
        elif bottleneck_type == 'gate_availability':
            # Gate delays can cascade to departure delays
            cascade_bottlenecks.append({
                'type': 'runway_departure_queue',
                'probability': severity * 0.6,
                'delay_minutes': initial_bottleneck.get('delay_minutes', 0) * 0.3,
                'cascade_source': 'gate_delay'
            })
        
        return cascade_bottlenecks
    
    def generate_mitigation_recommendations(self, bottleneck_data: Dict) -> List[Dict]:
        """
        Generate mitigation recommendations for bottlenecks
        
        Args:
            bottleneck_data: Bottleneck analysis data
            
        Returns:
            List of mitigation recommendations
        """
        recommendations = []
        bottleneck_type = bottleneck_data.get('type', 'unknown')
        severity = bottleneck_data.get('severity', 0.5)
        
        if bottleneck_type == 'runway_approach_queue':
            recommendations.extend([
                {
                    'action': 'Increase runway separation',
                    'priority': 'high' if severity > 0.7 else 'medium',
                    'estimated_effectiveness': 0.8,
                    'implementation_time': 5.0  # minutes
                },
                {
                    'action': 'Divert aircraft to alternate runways',
                    'priority': 'high' if severity > 0.8 else 'medium',
                    'estimated_effectiveness': 0.9,
                    'implementation_time': 10.0
                }
            ])
        
        elif bottleneck_type == 'taxiway_intersection':
            recommendations.extend([
                {
                    'action': 'Implement ground traffic sequencing',
                    'priority': 'medium',
                    'estimated_effectiveness': 0.7,
                    'implementation_time': 3.0
                },
                {
                    'action': 'Use alternative taxi routes',
                    'priority': 'low',
                    'estimated_effectiveness': 0.5,
                    'implementation_time': 2.0
                }
            ])
        
        elif bottleneck_type == 'gate_availability':
            recommendations.extend([
                {
                    'action': 'Reassign gates dynamically',
                    'priority': 'high' if severity > 0.6 else 'medium',
                    'estimated_effectiveness': 0.8,
                    'implementation_time': 5.0
                },
                {
                    'action': 'Use remote parking positions',
                    'priority': 'medium',
                    'estimated_effectiveness': 0.6,
                    'implementation_time': 8.0
                }
            ])
        
        return recommendations
