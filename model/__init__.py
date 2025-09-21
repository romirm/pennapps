"""
Air Traffic Bottleneck Prediction System

A comprehensive system that processes flight data from data.json and generates 
detailed bottleneck analysis reports using advanced clustering algorithms.
"""

# Import the main system components
from .main import AirTrafficBottleneckSystem
from .flight_processor import AirportDatabase, FlightProcessor, FlightPosition
from .bottleneck_analyzer import BottleneckAnalyzer, Bottleneck

__version__ = "1.0.0"
__author__ = "Air Traffic Bottleneck Prediction System"

__all__ = [
    "AirTrafficBottleneckSystem",
    "AirportDatabase",
    "FlightProcessor", 
    "FlightPosition",
    "BottleneckAnalyzer",
    "Bottleneck"
]