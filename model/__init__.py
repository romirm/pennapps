"""
Simplified MLP Airport Bottleneck Prediction System

A simple Multi-Layer Perceptron model that analyzes aircraft data from ADS-B.lol API 
to predict operational bottlenecks, resolution times, passenger impact, and fuel waste estimates.

Replaces the complex GNN-KAN hybrid with a straightforward neural network approach.
"""

# Import simplified components
from .simple_mlp_predictor import SimpleMLPPredictor, SimpleBottleneckPredictor, BottleneckPrediction
from .simple_airport_model import SimpleAirportBottleneckModel, AirportBottleneckModel
from .simple_config import SIMPLE_BOTTLENECK_CONFIG, BOTTLENECK_CONFIG, MODEL_CONFIG

# Import remaining components (still needed)
from .flight_metadata import FlightMetadataProcessor
from .impact_calculator import ImpactCalculator

# Backward compatibility imports
try:
    from .adsb_processor import ADSBDataProcessor
except ImportError:
    ADSBDataProcessor = None

try:
    from .bottleneck_gnn import BottleneckGNN
except ImportError:
    BottleneckGNN = None

try:
    from .kan_predictor import BottleneckKANPredictor
except ImportError:
    BottleneckKANPredictor = None

__version__ = "2.0.0"
__author__ = "Simplified Airport Bottleneck Prediction System"

__all__ = [
    # New simplified components
    "SimpleMLPPredictor",
    "SimpleBottleneckPredictor", 
    "BottleneckPrediction",
    "SimpleAirportBottleneckModel",
    "AirportBottleneckModel",
    
    # Configuration
    "SIMPLE_BOTTLENECK_CONFIG",
    "BOTTLENECK_CONFIG",
    "MODEL_CONFIG",
    
    # Remaining components
    "FlightMetadataProcessor",
    "ImpactCalculator",
    
    # Backward compatibility (if available)
    "ADSBDataProcessor",
    "BottleneckGNN",
    "BottleneckKANPredictor"
]
