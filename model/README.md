# GNN-KAN Airport Bottleneck Prediction System

A hybrid Graph Neural Network (GNN) + Kolmogorov-Arnold Network (KAN) model that analyzes aircraft data from ADS-B.lol API to predict operational bottlenecks, resolution times, passenger impact, and fuel waste estimates.

## 🎯 Core Mission

**INPUT**: Airport + nearby aircraft from ADS-B.lol  
**OUTPUT**: Bottleneck predictions with comprehensive impact analysis
- Where bottlenecks will likely occur
- How long they'll take to resolve  
- How many passengers affected
- Fuel waste estimates

## 🏗️ Architecture

### 1. ADS-B Data Processing Layer (`adsb_processor.py`)
- Loads and filters aircraft data from ADS-B.lol API (3nm radius)
- Identifies bottleneck zones (runway approaches, taxiway intersections, gate areas)
- Constructs spatial graphs for GNN analysis

### 2. GNN Bottleneck Analyzer (`bottleneck_gnn.py`)
- Models spatial relationships to identify congestion patterns
- Detects queue formation, intersection conflicts, gate congestion
- Provides bottleneck probability embeddings

### 3. KAN Impact Predictor (`kan_predictor.py`)
- Predicts bottleneck characteristics and impact metrics
- Estimates resolution time, passenger impact, fuel waste
- Generates mitigation recommendations

### 4. Flight Metadata Processor (`flight_metadata.py`)
- Enriches ADS-B data with aircraft characteristics
- Estimates passenger counts and fuel consumption
- Provides operational parameters for impact calculations

### 5. Impact Calculator (`impact_calculator.py`)
- Calculates economic, environmental, and operational impacts
- Provides cost-benefit analysis for mitigations
- Quantifies passenger experience metrics

### 6. Complete Model (`airport_bottleneck_model.py`)
- Integrates all components into unified system
- Provides comprehensive bottleneck analysis
- Generates actionable recommendations

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r model/requirements.txt

# Note: Some PyTorch Geometric dependencies may require specific installation
# See: https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
```

### Basic Usage

```python
from model import AirportBottleneckModel
from model.config import BOTTLENECK_CONFIG

# Initialize model
model = AirportBottleneckModel(BOTTLENECK_CONFIG)

# Load ADS-B data (example)
adsb_data = {
    "aircraft": [
        {
            "flight": "UAL123",
            "t": "B737",
            "lat": 40.6413,
            "lon": -73.7781,
            "alt_baro": 500,
            "track": 90,
            "gs": 150,
            "timestamp": "2024-01-01T12:00:00Z"
        }
        # ... more aircraft
    ]
}

# Airport configuration
airport_config = {
    "icao": "KJFK",
    "runways": ["09L/27R", "09R/27L"],
    "gates": ["A1", "A2", "B1", "B2"]
}

# Predict bottlenecks
analysis = model.predict_bottlenecks(adsb_data, airport_config)

# Access results
print(f"Total bottlenecks: {analysis['airport_summary']['total_bottlenecks_predicted']}")
print(f"Overall risk: {analysis['airport_summary']['overall_delay_risk']}")

for bottleneck in analysis['bottleneck_predictions']:
    print(f"Bottleneck: {bottleneck['type']}")
    print(f"Probability: {bottleneck['probability']:.2f}")
    print(f"Passengers affected: {bottleneck['impact_analysis']['passengers_affected']}")
    print(f"Fuel waste: {bottleneck['impact_analysis']['fuel_waste_gallons']:.1f} gallons")
```

## 📊 Output Format

The system provides comprehensive analysis in the following format:

```json
{
    "timestamp": "2024-01-01T12:00:00Z",
    "airport": "KJFK",
    "analysis_radius_nm": 3,
    "total_aircraft_monitored": 15,
    
    "bottleneck_predictions": [
        {
            "bottleneck_id": "runway_approach_09L",
            "location": {"zone": "runway_approach", "coordinates": [40.6413, -73.7781]},
            "type": "runway_approach_queue",
            "probability": 0.85,
            "severity": 4,
            
            "timing": {
                "predicted_onset_minutes": 0,
                "estimated_duration_minutes": 12.5,
                "resolution_confidence": 0.15
            },
            
            "aircraft_affected": [
                {
                    "flight_id": "UAL123",
                    "aircraft_type": "B737",
                    "estimated_passengers": 150,
                    "delay_contribution": 8.5,
                    "current_phase": "approach"
                }
            ],
            
            "impact_analysis": {
                "passengers_affected": 450,
                "total_delay_minutes": 12.5,
                "fuel_waste_gallons": 31.25,
                "fuel_cost_estimate": 109.38,
                "co2_emissions_lbs": 659.38,
                "economic_impact_estimate": 1406.25
            },
            
            "recommended_mitigations": [
                {
                    "action": "Increase runway separation",
                    "priority": "high",
                    "estimated_effectiveness": 0.8,
                    "implementation_time": 5.0
                }
            ]
        }
    ],
    
    "airport_summary": {
        "total_bottlenecks_predicted": 2,
        "highest_severity_level": 4,
        "total_passengers_at_risk": 450,
        "total_fuel_waste_estimate": 31.25,
        "overall_delay_risk": "high"
    }
}
```

## 🔧 Configuration

The system is highly configurable through `config.py`:

```python
BOTTLENECK_CONFIG = {
    # Detection thresholds
    'bottleneck_detection_threshold': 0.6,
    'queue_length_threshold': 3,
    
    # Model architecture
    'gnn_hidden_dim': 64,
    'gnn_layers': 4,
    
    # Economic parameters
    'fuel_price_per_gallon': 3.50,
    'passenger_compensation_rate': 2.50,
    
    # Monitoring
    'update_frequency_seconds': 30,
    'alert_severity_thresholds': {
        'critical': 0.9,
        'high': 0.7,
        'medium': 0.5,
        'low': 0.3
    }
}
```

## 🎯 Critical Bottleneck Zones

### 1. Runway Operations
- **Approach Queues**: Aircraft stacking on final approach
- **Departure Queues**: Aircraft waiting for takeoff clearance
- **Runway Crossings**: Aircraft waiting to cross active runways

### 2. Taxiway Intersections
- **Traffic Conflicts**: Multiple aircraft converging on taxi routes
- **Sequencing Delays**: First-come-first-served + priority rules
- **Cascading Effects**: Upstream congestion propagation

### 3. Gate and Terminal Areas
- **Gate Availability**: Aircraft waiting for gate assignment
- **Pushback Conflicts**: Aircraft unable to push back due to traffic
- **Remote Parking**: Holding patterns and remote parking delays

## 📈 Success Metrics

- **Prediction Accuracy**: >80% bottleneck prediction accuracy 15 minutes ahead
- **Impact Estimation**: ±20% accuracy on fuel waste and passenger impact
- **Operational Value**: Enable 30% reduction in bottleneck duration through early intervention
- **Economic Impact**: Quantify $millions in operational savings potential

## 🔮 Extension Points

### 1. Real-Time Monitoring Integration
```python
# TODO: Live ADS-B.lol API integration
class RealTimeBottleneckMonitor:
    def setup_adsb_stream(self, airports: List[str]): pass
    def detect_emerging_bottlenecks(self): pass
    def trigger_alerts(self, severity_threshold: float): pass
```

### 2. Historical Pattern Analysis
```python
# TODO: Learn from historical bottleneck patterns  
class HistoricalBottleneckAnalyzer:
    def analyze_seasonal_patterns(self): pass
    def identify_recurring_bottleneck_zones(self): pass
    def improve_prediction_accuracy(self): pass
```

### 3. Mitigation Recommendation Engine
```python
# TODO: Suggest operational interventions
class BottleneckMitigationEngine:
    def suggest_runway_usage_changes(self): pass
    def recommend_traffic_flow_adjustments(self): pass
    def estimate_intervention_effectiveness(self): pass
```

## 🛠️ Development Status

**Current Status**: Core model architecture implemented, ready for data integration

**Next Steps**:
1. Integrate with live ADS-B.lol API
2. Train on historical airport data
3. Implement real-time monitoring dashboard
4. Add weather integration for enhanced predictions
5. Develop ATC integration interface

## 📝 License

This project is part of the PennApps hackathon submission. See LICENSE file for details.

## 🤝 Contributing

This is a hackathon project. For production use, additional testing, validation, and optimization would be required.

---

**Built for PennApps 2025** - Airport Bottleneck Prediction System
