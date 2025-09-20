"""
Simplified Configuration for MLP-based Airport Bottleneck Prediction System
"""

# Simplified Bottleneck Analysis Configuration
SIMPLE_BOTTLENECK_CONFIG = {
    # Data Processing
    'adsb_radius_nm': 3,
    'update_frequency_seconds': 30,
    'bottleneck_detection_threshold': 0.6,
    
    # MLP Architecture (replaces GNN-KAN)
    'mlp_input_dim': 20,           # Number of input features
    'mlp_hidden_dim': 128,          # Hidden layer size
    'mlp_output_dim': 8,            # Output dimensions (4 zones + 4 impact metrics)
    
    # Bottleneck Thresholds
    'queue_length_threshold': 3,      # Aircraft count triggering bottleneck
    'delay_threshold_minutes': 5,     # Delay threshold for impact calculation  
    'fuel_waste_threshold_gallons': 50,  # Fuel waste threshold for alerts
    
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
    
    # Aircraft Type Parameters
    'aircraft_types': {
        'B737': {'passengers': 150, 'fuel_burn_rate': 2.5},
        'A320': {'passengers': 160, 'fuel_burn_rate': 2.3},
        'B777': {'passengers': 300, 'fuel_burn_rate': 5.0},
        'A380': {'passengers': 500, 'fuel_burn_rate': 8.0},
        'B787': {'passengers': 250, 'fuel_burn_rate': 3.5},
        'A350': {'passengers': 280, 'fuel_burn_rate': 3.2},
        'DEFAULT': {'passengers': 150, 'fuel_burn_rate': 2.5}
    },
    
    # Bottleneck Zone Definitions
    'bottleneck_zones': {
        'runway_approach': {
            'radius_nm': 2.0,
            'density_threshold': 0.8,
            'priority': 'critical'
        },
        'taxiway_intersection': {
            'radius_nm': 0.5,
            'density_threshold': 0.6,
            'priority': 'high'
        },
        'gate_area': {
            'radius_nm': 1.0,
            'density_threshold': 0.4,
            'priority': 'medium'
        },
        'departure_queue': {
            'radius_nm': 1.5,
            'density_threshold': 0.7,
            'priority': 'high'
        }
    },
    
    # Alert Severity Thresholds
    'alert_severity_thresholds': {
        'critical': 0.9,
        'high': 0.7,
        'medium': 0.5,
        'low': 0.3
    },
    
    # Model Training Parameters
    'training': {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'validation_split': 0.2,
        'early_stopping_patience': 10
    },
    
    # Real-time Monitoring
    'monitoring': {
        'update_interval_seconds': 30,
        'max_aircraft_per_analysis': 50,
        'min_aircraft_for_prediction': 1,
        'prediction_horizon_minutes': 15
    },
    
    # Performance Metrics
    'performance_targets': {
        'prediction_accuracy_threshold': 0.75,
        'false_positive_rate_threshold': 0.2,
        'response_time_threshold_ms': 1000
    }
}

# Backward compatibility - use simple config as default
BOTTLENECK_CONFIG = SIMPLE_BOTTLENECK_CONFIG

# Model Architecture Configuration
MODEL_CONFIG = {
    'model_type': 'Simple MLP',
    'version': '1.0',
    'description': 'Simplified Multi-Layer Perceptron replacing GNN-KAN hybrid',
    
    'architecture': {
        'input_layer': {
            'size': 20,
            'description': 'Aircraft features (count, speed, altitude, position, type)'
        },
        'hidden_layers': [
            {
                'size': 128,
                'activation': 'ReLU',
                'dropout': 0.2
            },
            {
                'size': 128,
                'activation': 'ReLU', 
                'dropout': 0.2
            },
            {
                'size': 64,
                'activation': 'ReLU',
                'dropout': 0.1
            }
        ],
        'output_layer': {
            'size': 8,
            'activation': 'Sigmoid',
            'description': '4 bottleneck probabilities + 4 impact metrics'
        }
    },
    
    'features': [
        'aircraft_count',
        'avg_speed', 'speed_std', 'max_speed',
        'avg_altitude', 'altitude_std',
        'avg_distance_from_airport', 'distance_std', 'max_distance',
        'b737_ratio', 'a320_ratio', 'b777_ratio', 'other_ratio',
        'avg_track', 'track_std',
        'hour_of_day', 'day_of_week', 'season'
    ],
    
    'outputs': [
        'runway_approach_probability',
        'taxiway_intersection_probability', 
        'gate_area_probability',
        'departure_queue_probability',
        'estimated_delay_minutes',
        'passengers_affected',
        'fuel_waste_gallons',
        'severity_level'
    ]
}

# Airport-specific configurations
AIRPORT_CONFIGS = {
    'KJFK': {
        'name': 'John F. Kennedy International Airport',
        'coordinates': (40.63980103, -73.77890015),
        'runways': ['04L/22R', '04R/22L', '09L/27R', '09R/27L', '13L/31R'],
        'gates': ['A1-A20', 'B1-B20', 'C1-C20', 'D1-D20'],
        'capacity': {
            'max_aircraft': 100,
            'peak_hourly_operations': 80
        }
    },
    'KLAX': {
        'name': 'Los Angeles International Airport',
        'coordinates': (33.9425, -118.4081),
        'runways': ['06L/24R', '06R/24L', '07L/25R', '07R/25L'],
        'gates': ['T1-T8', 'TBIT'],
        'capacity': {
            'max_aircraft': 120,
            'peak_hourly_operations': 100
        }
    },
    'KPHL': {
        'name': 'Philadelphia International Airport',
        'coordinates': (39.8719, -75.2411),
        'runways': ['08/26', '09L/27R', '09R/27L', '17/35'],
        'gates': ['A1-A15', 'B1-B15', 'C1-C15', 'D1-D15'],
        'capacity': {
            'max_aircraft': 80,
            'peak_hourly_operations': 60
        }
    }
}

# Default configuration for unknown airports
DEFAULT_AIRPORT_CONFIG = {
    'name': 'Unknown Airport',
    'coordinates': (0.0, 0.0),
    'runways': ['09/27', '18/36'],
    'gates': ['A1-A10'],
    'capacity': {
        'max_aircraft': 50,
        'peak_hourly_operations': 40
    }
}
