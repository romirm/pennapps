"""
Simple Bottleneck Predictor

A lightweight bottleneck prediction system that analyzes aircraft traffic patterns
and predicts potential bottlenecks using either the Cerebras API or simple heuristic algorithms.
"""

# Import the simple bottleneck predictor
from .simple_bottleneck_predictor import SimpleBottleneckPredictor

__version__ = "3.0.0"
__author__ = "Simple Bottleneck Prediction System"

__all__ = [
    "SimpleBottleneckPredictor"
]