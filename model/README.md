# Simple Bottleneck Predictor

A lightweight bottleneck prediction system that analyzes aircraft traffic patterns and predicts potential bottlenecks using either the Cerebras API or simple heuristic algorithms.

## Features

- **Real-time Analysis**: Processes live aircraft data from the main application
- **Cerebras Integration**: Uses Cerebras API for advanced bottleneck prediction when available
- **Fallback Mode**: Simple heuristic-based prediction when Cerebras is not available
- **Automatic Logging**: Saves prediction results to `results.txt`
- **Traffic Density Analysis**: Analyzes ground, low-altitude, and high-altitude traffic patterns
- **Hotspot Detection**: Identifies areas with high aircraft concentration

## Files

- `simple_bottleneck_predictor.py`: Main predictor class with Cerebras integration
- `results.txt`: Output file containing prediction results and analysis
- `requirements.txt`: Dependencies for the predictor

## Usage

The predictor is automatically integrated into the main Flask application and runs every 30 seconds to analyze current aircraft traffic patterns.

### Manual Usage

```python
from model.simple_bottleneck_predictor import SimpleBottleneckPredictor

predictor = SimpleBottleneckPredictor()

# Sample aircraft data
aircraft_data = [
    {
        "flight": "DAL123",
        "lat": 40.6413,
        "lon": -73.7781,
        "altitude": "ground",
        "speed": 0,
        "heading": 90
    }
]

# Predict bottlenecks
results = predictor.predict_and_save(aircraft_data, "JFK")
```

## Output Format

Results are saved to `results.txt` with:
- Bottleneck likelihood percentage
- Risk level (Low/Medium/High)
- Traffic analysis metrics
- Recommendations
- Cerebras analysis (when available)

## Dependencies

- `cerebras-cloud-sdk`: For advanced AI-powered predictions
- `flask`: Web framework integration
- `aiohttp`: Async HTTP client
- `requests`: HTTP requests