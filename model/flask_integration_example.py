"""
Flask Integration Example for Bottleneck Prediction

This shows you exactly how to integrate the simplified MLP model
into your existing Flask app to show bottleneck predictions on your map.
"""

from flask import Flask, render_template, jsonify, request
import json
from datetime import datetime

# Import the simplified model
from simple_airport_model import SimpleAirportBottleneckModel
from simple_config import SIMPLE_BOTTLENECK_CONFIG

app = Flask(__name__)

# Initialize the model once (for efficiency)
bottleneck_model = SimpleAirportBottleneckModel(SIMPLE_BOTTLENECK_CONFIG)

def get_airport_coordinates(airport_code):
    """Get airport coordinates"""
    airport_coords = {
        'KJFK': (40.63980103, -73.77890015),
        'KLAX': (33.9425, -118.4081),
        'KORD': (41.9786, -87.9048),
        'KDFW': (32.8968, -97.0380),
        'KPHL': (39.8719, -75.2411),
        'KATL': (33.6407, -84.4277),
        'KLGA': (40.7769, -73.8740),
        'KBOS': (42.3656, -71.0096)
    }
    return airport_coords.get(airport_code, (0.0, 0.0))

def create_sample_aircraft_data(airport_code):
    """Create sample aircraft data for demonstration"""
    airport_coords = get_airport_coordinates(airport_code)
    
    # Sample aircraft near the airport
    sample_aircraft = [
        {
            "flight": "UAL123",
            "t": "B737",
            "lat": airport_coords[0] + 0.0015,  # Slightly north
            "lon": airport_coords[1] - 0.0003,  # Slightly west
            "alt_baro": 500,
            "track": 90,
            "gs": 150,
            "timestamp": datetime.now().isoformat()
        },
        {
            "flight": "DAL456",
            "t": "A320", 
            "lat": airport_coords[0] - 0.0008,  # Slightly south
            "lon": airport_coords[1] + 0.0002,  # Slightly east
            "alt_baro": 600,
            "track": 85,
            "gs": 140,
            "timestamp": datetime.now().isoformat()
        },
        {
            "flight": "SWA789",
            "t": "B737",
            "lat": airport_coords[0] + 0.0002,  # Very close
            "lon": airport_coords[1] - 0.0001,  # Very close
            "alt_baro": 400,
            "track": 95,
            "gs": 120,
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    return {"aircraft": sample_aircraft}

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/airport/<code>')
def airport(code):
    """Airport page with map"""
    airport_code = code.upper()
    
    # Get airport coordinates
    coords = get_airport_coordinates(airport_code)
    
    # Create sample airport data
    airport_data = {
        'name': f'{airport_code} Airport',
        'city': 'Unknown',
        'country': 'Unknown',
        'coordinates': coords,
        'runways': [
            {'id': '09/27', 'length': 3000, 'width': 45, 'heading': 90},
            {'id': '18/36', 'length': 2500, 'width': 45, 'heading': 180}
        ],
        'taxiways': []
    }
    
    return render_template('airport.html', 
                         airport_code=airport_code, 
                         airport=airport_data)

@app.route('/api/bottlenecks/<code>')
def get_bottlenecks(code):
    """
    API endpoint to get bottleneck predictions
    
    This is where you feed data into the model and get predictions!
    """
    airport_code = code.upper()
    
    try:
        # 1. CREATE SAMPLE AIRCRAFT DATA
        # In production, you would fetch real data from ADS-B.lol API
        adsb_data = create_sample_aircraft_data(airport_code)
        
        # 2. CREATE AIRPORT CONFIGURATION
        airport_config = {
            'icao': airport_code,
            'coordinates': get_airport_coordinates(airport_code)
        }
        
        # 3. FEED DATA INTO MODEL AND GET PREDICTIONS
        print(f"🔍 Analyzing {len(adsb_data['aircraft'])} aircraft at {airport_code}")
        analysis = bottleneck_model.predict_bottlenecks(adsb_data, airport_config)
        
        # 4. RETURN PREDICTIONS
        return jsonify({
            'success': True,
            'airport': airport_code,
            'timestamp': datetime.now().isoformat(),
            'aircraft_count': len(adsb_data['aircraft']),
            'bottlenecks': analysis['bottleneck_predictions'],
            'summary': analysis['airport_summary']
        })
        
    except Exception as e:
        print(f"❌ Error predicting bottlenecks: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'airport': airport_code
        })

@app.route('/api/aircraft/<code>')
def get_aircraft_data(code):
    """
    API endpoint to get current aircraft data
    
    This shows you the raw data that gets fed into the model
    """
    airport_code = code.upper()
    
    try:
        # Create sample aircraft data
        aircraft_data = create_sample_aircraft_data(airport_code)
        
        return jsonify({
            'success': True,
            'airport': airport_code,
            'timestamp': datetime.now().isoformat(),
            'aircraft': aircraft_data['aircraft']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    print("🚀 Starting Flask app with Bottleneck Prediction")
    print("📍 Available endpoints:")
    print("   • /airport/<code> - Airport map page")
    print("   • /api/bottlenecks/<code> - Get bottleneck predictions")
    print("   • /api/aircraft/<code> - Get aircraft data")
    print("\n🎯 Example usage:")
    print("   • http://localhost:5000/airport/JFK")
    print("   • http://localhost:5000/api/bottlenecks/JFK")
    print("   • http://localhost:5000/api/aircraft/JFK")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
