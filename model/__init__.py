"""
Simple Bottleneck Predictor

A lightweight bottleneck prediction system that analyzes aircraft traffic patterns
and predicts potential bottlenecks using either the Cerebras API or simple heuristic algorithms.
"""

# Import the agentic AI bottleneck predictor
from .agentic_bottleneck_predictor import AgenticBottleneckPredictor, AgenticCerebrasAnalyzer, AircraftDatabase, AircraftInfo, BottleneckImpact, PilotCommunication

__version__ = "5.0.0"
__author__ = "Agentic AI Bottleneck Prediction System with Comprehensive Analysis"

__all__ = [
    "AgenticBottleneckPredictor",
    "AgenticCerebrasAnalyzer", 
    "AircraftDatabase",
    "AircraftInfo",
    "BottleneckImpact",
    "PilotCommunication"
]