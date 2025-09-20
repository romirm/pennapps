# Testing the GNN-KAN Airport Bottleneck Prediction System

This directory contains tools to test the bottleneck prediction system with real flight data.

## 🚀 Quick Start

### 1. Test with Sample Data (Recommended for first run)

```bash
# Run the demo with sample data
python model/demo.py

# Or run the interactive tool
python model/interactive_predictor.py

# Or run the test script
python model/test_bottleneck_prediction.py --sample
```

### 2. Test with Real Flight Data

```bash
# Test with specific flights
python model/test_bottleneck_prediction.py --flights UAL123 DAL456 SWA789 --airport KJFK

# Test with all aircraft near an airport
python model/test_bottleneck_prediction.py --airport KJFK --radius 5

# Interactive mode
python model/interactive_predictor.py
```

## 📋 Available Tools

### 1. `demo.py` - Complete Demo
- Shows both sample data and real flight data examples
- Interactive choice between demo types
- Detailed result display

### 2. `interactive_predictor.py` - Interactive Tool
- Menu-driven interface
- Test specific flights or airport areas
- Save results automatically

### 3. `test_bottleneck_prediction.py` - Command Line Tool
- Command line interface
- Batch processing capabilities
- Flexible parameter options

### 4. `flight_bottleneck_predictor.py` - Core Prediction Engine
- Main prediction logic
- ADS-B.lol API integration
- Real-time data fetching

## 🛫 How to Use with Real Flights

### Step 1: Find Active Flights
1. Go to [FlightAware](https://flightaware.com) or [FlightRadar24](https://flightradar24.com)
2. Search for flights at your target airport (e.g., KJFK)
3. Note down active flight numbers (e.g., UAL123, DAL456)

### Step 2: Run Prediction
```bash
# Example with real flights
python model/test_bottleneck_prediction.py --flights UAL123 DAL456 SWA789 --airport KJFK
```

### Step 3: Analyze Results
- Results are saved as JSON files with timestamps
- Check the console output for immediate analysis
- Use the JSON files for detailed analysis

## 📊 Understanding the Output

### Airport Summary
- **Total aircraft monitored**: Number of aircraft analyzed
- **Bottlenecks predicted**: Number of bottlenecks detected
- **Highest severity level**: 1-5 scale (5 = critical)
- **Overall delay risk**: low/medium/high/critical
- **Passengers at risk**: Total passengers affected
- **Fuel waste estimate**: Gallons of fuel wasted

### Bottleneck Details
- **Type**: runway_approach_queue, taxiway_intersection, gate_availability, etc.
- **Probability**: 0-1 scale (likelihood of bottleneck)
- **Severity**: 1-5 scale (impact level)
- **Duration**: Estimated duration in minutes
- **Aircraft affected**: List of affected flights
- **Impact analysis**: Passengers, fuel, cost, emissions
- **Recommended mitigations**: Suggested actions

## 🔧 Configuration

The system uses `config.py` for configuration. Key parameters:

```python
BOTTLENECK_CONFIG = {
    'bottleneck_detection_threshold': 0.6,  # Sensitivity
    'gnn_hidden_dim': 64,                   # Model complexity
    'fuel_price_per_gallon': 3.50,         # Economic calculations
    'passenger_compensation_rate': 2.50,   # Cost per passenger per minute
}
```

## 🌐 Supported Airports

The system supports major airports worldwide:

**US Airports:**
- KJFK (New York JFK)
- KLAX (Los Angeles)
- KPHL (Philadelphia)
- KMIA (Miami)
- KORD (Chicago O'Hare)
- KDFW (Dallas/Fort Worth)
- KATL (Atlanta)
- KDEN (Denver)
- KSFO (San Francisco)
- KBOS (Boston)
- KSEA (Seattle)
- KLAS (Las Vegas)
- KMCO (Orlando)

**International Airports:**
- LHR (London Heathrow)
- CDG (Paris Charles de Gaulle)
- NRT (Tokyo Narita)
- ICN (Seoul Incheon)
- DXB (Dubai)
- SIN (Singapore)
- HKG (Hong Kong)

## 📡 Data Sources

The system fetches real-time data from:
- **ADS-B.lol API**: Primary source for aircraft positions
- **Flight metadata**: Aircraft types, passenger counts, fuel consumption
- **Airport databases**: Runway configurations, gate layouts

## ⚠️ Important Notes

1. **Flight Numbers**: Use real, active flight numbers for meaningful results
2. **API Limits**: ADS-B.lol has rate limits - the system includes delays
3. **Data Availability**: Not all flights may be visible in ADS-B data
4. **Model Training**: The current model needs training data for optimal performance

## 🐛 Troubleshooting

### No Aircraft Data Found
- Check if flight numbers are correct and active
- Verify airport ICAO code is valid
- Try increasing search radius

### API Errors
- Check internet connection
- ADS-B.lol may be temporarily unavailable
- Try again in a few minutes

### Model Errors
- This is expected for the initial implementation
- The model architecture is ready but needs training data
- Sample data tests should work for demonstration

## 📈 Expected Results

### Sample Data Test
- Should complete without errors
- Shows model architecture working
- Demonstrates output format

### Real Flight Data Test
- May show "no data" if flights aren't active
- Will fetch real aircraft positions when available
- Provides actual bottleneck predictions

## 🔮 Next Steps

1. **Integrate with live data streams**
2. **Train model on historical data**
3. **Add weather integration**
4. **Implement real-time dashboard**
5. **Connect to ATC systems**

---

**Built for PennApps 2025** - Airport Bottleneck Prediction System
