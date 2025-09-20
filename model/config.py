"""
Configuration template for GNN-KAN Airport Bottleneck Prediction System
"""

# Bottleneck Analysis Configuration
BOTTLENECK_CONFIG = {
    # Data Processing
    'adsb_radius_nm': 3,
    'update_frequency_seconds': 30,
    'bottleneck_detection_threshold': 0.6,
    
    # GNN Architecture
    'spatial_resolution_meters': 100,  # Grid resolution for bottleneck zones
    'gnn_input_dim': 5,
    'gnn_hidden_dim': 64,
    'gnn_layers': 4,
    'gnn_output_dim': 32,
    
    # KAN Prediction
    'flight_metadata_dim': 10,
    'kan_output_dim': 64,
    'impact_prediction_horizon_minutes': 60,
    
    # Bottleneck Thresholds
    'queue_length_threshold': 3,      # Aircraft count triggering bottleneck
    'delay_threshold_minutes': 5,     # Delay threshold for impact calculation  
    'fuel_waste_threshold_gallons': 100,  # Fuel waste threshold for alerts
    
    # Economic Parameters
    'passenger_compensation_rate': 2.50,  # USD per passenger per minute
    'fuel_price_per_gallon': 3.50,       # USD
    'co2_cost_per_ton': 50.0,            # USD (carbon pricing)
    'airline_reputation_cost': 1000.0,   # USD per incident
    
    # Environmental Parameters
    'co2_per_gallon_fuel': 21.1,         # lbs CO2 per gallon
    'co2_per_ton': 2204.62,              # lbs per ton
    
    # Operational Parameters
    'on_time_performance_weight': 0.3,
    'customer_satisfaction_weight': 0.4,
    'operational_cost_weight': 0.3,
    
    # Load Factor Assumptions
    'passenger_load_factors': {
        'domestic': 0.85,
        'international': 0.80,
        'regional': 0.75,
        'cargo': 0.0
    },
    
    # Fuel Burn Rates (gallons per hour)
    'fuel_burn_rates': {
        'B737': {'idle': 150, 'holding': 225},
        'A320': {'idle': 140, 'holding': 210},
        'B777': {'idle': 300, 'holding': 450},
        'A380': {'idle': 500, 'holding': 750},
        'CRJ9': {'idle': 80, 'holding': 120},
        'E175': {'idle': 75, 'holding': 112},
        'B767F': {'idle': 250, 'holding': 375}
    },
    
    # Bottleneck Zone Definitions
    'bottleneck_zones': {
        'runway_approach_queue': {
            'description': 'Aircraft stacking on final approach',
            'typical_capacity': 3,
            'bottleneck_triggers': ['weather', 'spacing_violations', 'go_arounds'],
            'impact_multiplier': 'high',
            'base_resolution_time': 5  # minutes per additional aircraft
        },
        'runway_departure_queue': {
            'description': 'Aircraft waiting for takeoff clearance', 
            'typical_capacity': 2,
            'bottleneck_triggers': ['traffic_spacing', 'weather', 'runway_crossings'],
            'impact_multiplier': 'high',
            'base_resolution_time': 3  # minutes per additional aircraft
        },
        'taxiway_intersection': {
            'description': 'Aircraft waiting to cross active runways',
            'typical_capacity': 1,
            'bottleneck_triggers': ['runway_crossings', 'traffic_conflicts'],
            'impact_multiplier': 'medium',
            'base_resolution_time': 2  # minutes per conflict
        },
        'gate_availability': {
            'description': 'Aircraft waiting for gate assignment',
            'typical_capacity': 8,
            'bottleneck_triggers': ['gate_blocking', 'pushback_conflicts'],
            'impact_multiplier': 'high',
            'base_resolution_time': 15  # minutes average gate wait
        }
    },
    
    # Mitigation Options
    'mitigation_options': {
        'runway_approach_queue': [
            {
                'action': 'Increase runway separation',
                'implementation_time': 5.0,
                'estimated_effectiveness': 0.8,
                'implementation_cost': 0
            },
            {
                'action': 'Divert aircraft to alternate runways',
                'implementation_time': 10.0,
                'estimated_effectiveness': 0.9,
                'implementation_cost': 5000
            },
            {
                'action': 'Implement ground delay program',
                'implementation_time': 15.0,
                'estimated_effectiveness': 0.7,
                'implementation_cost': 2000
            }
        ],
        'runway_departure_queue': [
            {
                'action': 'Optimize departure sequencing',
                'implementation_time': 3.0,
                'estimated_effectiveness': 0.6,
                'implementation_cost': 0
            },
            {
                'action': 'Use multiple departure runways',
                'implementation_time': 8.0,
                'estimated_effectiveness': 0.8,
                'implementation_cost': 3000
            }
        ],
        'taxiway_intersection': [
            {
                'action': 'Implement ground traffic sequencing',
                'implementation_time': 3.0,
                'estimated_effectiveness': 0.7,
                'implementation_cost': 1000
            },
            {
                'action': 'Use alternative taxi routes',
                'implementation_time': 2.0,
                'estimated_effectiveness': 0.5,
                'implementation_cost': 500
            }
        ],
        'gate_availability': [
            {
                'action': 'Reassign gates dynamically',
                'implementation_time': 5.0,
                'estimated_effectiveness': 0.8,
                'implementation_cost': 2000
            },
            {
                'action': 'Use remote parking positions',
                'implementation_time': 8.0,
                'estimated_effectiveness': 0.6,
                'implementation_cost': 1500
            },
            {
                'action': 'Implement gate sharing',
                'implementation_time': 10.0,
                'estimated_effectiveness': 0.7,
                'implementation_cost': 3000
            }
        ]
    },
    
    # Success Metrics Thresholds
    'success_metrics': {
        'prediction_accuracy_threshold': 0.8,  # 80% accuracy target
        'impact_estimation_accuracy': 0.2,     # ±20% accuracy on fuel/passenger impact
        'operational_savings_target': 0.3,     # 30% reduction in bottleneck duration
        'economic_savings_threshold': 1000000  # $1M operational savings potential
    },
    
    # Real-time Monitoring
    'monitoring': {
        'alert_severity_thresholds': {
            'low': 0.3,
            'medium': 0.5,
            'high': 0.7,
            'critical': 0.9
        },
        'update_intervals': {
            'low_severity': 60,    # seconds
            'medium_severity': 30,
            'high_severity': 15,
            'critical_severity': 5
        }
    }
}

# Example usage configuration
EXAMPLE_CONFIG = {
    'airport_icao': 'KJFK',
    'analysis_radius_nm': 3,
    'bottleneck_detection_threshold': 0.6,
    'gnn_hidden_dim': 64,
    'gnn_layers': 4,
    'impact_prediction_horizon_minutes': 60,
    'fuel_price_per_gallon': 3.50,
    'passenger_compensation_rate': 2.50
}
