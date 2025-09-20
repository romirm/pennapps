"""
Interactive Bottleneck Prediction Demo

This script shows you exactly where to feed data into the model
and how to visualize where bottlenecks are predicted to occur.
"""

import json
import sys
import os
from datetime import datetime
import requests

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_mlp_predictor import SimpleBottleneckPredictor, BottleneckPrediction
from simple_airport_model import SimpleAirportBottleneckModel
from simple_config import SIMPLE_BOTTLENECK_CONFIG


def fetch_real_adsb_data(airport_code="KJFK"):
    """
    Fetch real aircraft data from ADS-B.lol API
    
    This is where you feed REAL data into the model!
    """
    print(f"🛩️  Fetching real aircraft data for {airport_code}...")
    
    try:
        # Get airport coordinates first
        airport_coords = get_airport_coordinates(airport_code)
        
        # ADS-B.lol API endpoint
        url = f"https://adsb.lol/api/aircraft/lat/{airport_coords[0]}/lon/{airport_coords[1]}/dist/3"
        
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Found {len(data.get('aircraft', []))} aircraft near {airport_code}")
            return data
        else:
            print(f"❌ API request failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None


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


def create_sample_data_scenarios():
    """
    Create different scenarios to test bottleneck prediction
    
    This shows you how to feed DIFFERENT TYPES of data into the model
    """
    scenarios = {
        "low_traffic": {
            "aircraft": [
                {
                    "flight": "UAL123",
                    "t": "B737",
                    "lat": 40.6413,
                    "lon": -73.7781,
                    "alt_baro": 500,
                    "track": 90,
                    "gs": 150
                }
            ]
        },
        
        "moderate_traffic": {
            "aircraft": [
                {
                    "flight": "UAL123", "t": "B737", "lat": 40.6413, "lon": -73.7781,
                    "alt_baro": 500, "track": 90, "gs": 150
                },
                {
                    "flight": "DAL456", "t": "A320", "lat": 40.6405, "lon": -73.7778,
                    "alt_baro": 600, "track": 85, "gs": 140
                },
                {
                    "flight": "SWA789", "t": "B737", "lat": 40.6398, "lon": -73.7795,
                    "alt_baro": 400, "track": 95, "gs": 120
                }
            ]
        },
        
        "high_traffic": {
            "aircraft": [
                {
                    "flight": "UAL123", "t": "B737", "lat": 40.6413, "lon": -73.7781,
                    "alt_baro": 500, "track": 90, "gs": 150
                },
                {
                    "flight": "DAL456", "t": "A320", "lat": 40.6405, "lon": -73.7778,
                    "alt_baro": 600, "track": 85, "gs": 140
                },
                {
                    "flight": "SWA789", "t": "B737", "lat": 40.6398, "lon": -73.7795,
                    "alt_baro": 400, "track": 95, "gs": 120
                },
                {
                    "flight": "JBU012", "t": "A320", "lat": 40.6410, "lon": -73.7785,
                    "alt_baro": 550, "track": 88, "gs": 135
                },
                {
                    "flight": "AA345", "t": "B777", "lat": 40.6402, "lon": -73.7772,
                    "alt_baro": 700, "track": 92, "gs": 160
                },
                {
                    "flight": "DL678", "t": "A321", "lat": 40.6408, "lon": -73.7788,
                    "alt_baro": 450, "track": 87, "gs": 125
                },
                {
                    "flight": "WN901", "t": "B737", "lat": 40.6400, "lon": -73.7775,
                    "alt_baro": 520, "track": 91, "gs": 145
                }
            ]
        },
        
        "runway_approach_congestion": {
            "aircraft": [
                # Multiple aircraft approaching same runway
                {"flight": "UAL123", "t": "B737", "lat": 40.6413, "lon": -73.7781, "alt_baro": 500, "track": 90, "gs": 150},
                {"flight": "DAL456", "t": "A320", "lat": 40.6415, "lon": -73.7783, "alt_baro": 600, "track": 90, "gs": 140},
                {"flight": "SWA789", "t": "B737", "lat": 40.6417, "lon": -73.7785, "alt_baro": 700, "track": 90, "gs": 130},
                {"flight": "JBU012", "t": "A320", "lat": 40.6419, "lon": -73.7787, "alt_baro": 800, "track": 90, "gs": 120}
            ]
        },
        
        "taxiway_intersection_conflict": {
            "aircraft": [
                # Aircraft converging on taxiway intersection
                {"flight": "UAL123", "t": "B737", "lat": 40.6405, "lon": -73.7778, "alt_baro": 0, "track": 90, "gs": 20},
                {"flight": "DAL456", "t": "A320", "lat": 40.6403, "lon": -73.7776, "alt_baro": 0, "track": 180, "gs": 25},
                {"flight": "SWA789", "t": "B737", "lat": 40.6407, "lon": -73.7780, "alt_baro": 0, "track": 270, "gs": 18}
            ]
        }
    }
    
    return scenarios


def visualize_bottleneck_locations(predictions, airport_code):
    """
    Show exactly WHERE bottlenecks are predicted to occur
    
    This is where you see the PREDICTED LOCATIONS!
    """
    print(f"\n🗺️  BOTTLENECK LOCATION MAP for {airport_code}")
    print("=" * 50)
    
    airport_coords = get_airport_coordinates(airport_code)
    print(f"📍 Airport Center: {airport_coords[0]:.4f}, {airport_coords[1]:.4f}")
    
    if not predictions:
        print("✅ No bottlenecks predicted - all areas clear!")
        return
    
    for i, bottleneck in enumerate(predictions, 1):
        # Handle both dict and object formats
        if isinstance(bottleneck, dict):
            zone_type = bottleneck.get('type', 'unknown')
            coordinates = bottleneck.get('location', {}).get('coordinates', airport_coords)
            probability = bottleneck.get('probability', 0)
            severity = bottleneck.get('severity', 0)
            delay_minutes = bottleneck.get('timing', {}).get('estimated_duration_minutes', 0)
            passengers = bottleneck.get('impact_analysis', {}).get('passengers_affected', 0)
            fuel_waste = bottleneck.get('impact_analysis', {}).get('fuel_waste_gallons', 0)
        else:
            zone_type = bottleneck.zone_type
            coordinates = bottleneck.coordinates
            probability = bottleneck.probability
            severity = bottleneck.severity
            delay_minutes = bottleneck.estimated_delay_minutes
            passengers = bottleneck.passengers_affected
            fuel_waste = bottleneck.fuel_waste_gallons
        
        print(f"\n🚨 BOTTLENECK #{i}: {zone_type.upper()}")
        print(f"   📍 Location: {coordinates[0]:.4f}, {coordinates[1]:.4f}")
        print(f"   📊 Probability: {probability:.2f}")
        print(f"   ⚠️  Severity: {severity}/5")
        print(f"   ⏱️  Estimated Delay: {delay_minutes:.1f} minutes")
        print(f"   👥 Passengers Affected: {passengers}")
        print(f"   ⛽ Fuel Waste: {fuel_waste:.1f} gallons")
        
        # Show relative position from airport center
        lat_diff = coordinates[0] - airport_coords[0]
        lon_diff = coordinates[1] - airport_coords[1]
        print(f"   📏 Distance from airport: {lat_diff:.4f}° lat, {lon_diff:.4f}° lon")


def run_scenario_test(scenario_name, adsb_data, airport_code):
    """Test a specific scenario and show results"""
    print(f"\n🧪 TESTING SCENARIO: {scenario_name.upper()}")
    print("=" * 60)
    
    # Initialize model
    model = SimpleAirportBottleneckModel(SIMPLE_BOTTLENECK_CONFIG)
    
    # Create airport config
    airport_config = {
        "icao": airport_code,
        "coordinates": get_airport_coordinates(airport_code)
    }
    
    # Feed data into model and get predictions
    print(f"📊 Feeding {len(adsb_data['aircraft'])} aircraft into model...")
    analysis = model.predict_bottlenecks(adsb_data, airport_config)
    
    # Show where bottlenecks are predicted
    predictions = analysis['bottleneck_predictions']
    visualize_bottleneck_locations(predictions, airport_code)
    
    # Show summary
    summary = analysis['airport_summary']
    print(f"\n📈 SUMMARY:")
    print(f"   • Total Bottlenecks: {summary['total_bottlenecks_predicted']}")
    print(f"   • Risk Level: {summary['overall_delay_risk'].upper()}")
    print(f"   • Passengers at Risk: {summary['total_passengers_at_risk']}")
    print(f"   • Fuel Waste: {summary['total_fuel_waste_estimate']:.1f} gallons")
    
    return analysis


def main():
    """Main interactive demo"""
    print("🎯 INTERACTIVE BOTTLENECK PREDICTION DEMO")
    print("=" * 60)
    print("This shows you EXACTLY where to feed data and see predictions!")
    
    # 1. Test with sample scenarios
    print("\n1️⃣ TESTING WITH SAMPLE SCENARIOS")
    scenarios = create_sample_data_scenarios()
    
    for scenario_name, adsb_data in scenarios.items():
        run_scenario_test(scenario_name, adsb_data, "KJFK")
        print("\n" + "─" * 60)
    
    # 2. Test with real data (if available)
    print("\n2️⃣ TESTING WITH REAL ADS-B DATA")
    print("🛩️  Attempting to fetch real aircraft data...")
    
    real_data = fetch_real_adsb_data("KJFK")
    if real_data and real_data.get('aircraft'):
        run_scenario_test("REAL_DATA", real_data, "KJFK")
    else:
        print("❌ Could not fetch real data (API might be down)")
        print("💡 Using sample data instead...")
        run_scenario_test("SAMPLE_DATA", scenarios["moderate_traffic"], "KJFK")
    
    # 3. Show how to integrate with your Flask app
    print("\n3️⃣ INTEGRATION WITH YOUR FLASK APP")
    print("=" * 60)
    print("""
    # In your app.py, add this route:
    
    @app.route('/airport/<code>/bottlenecks')
    def get_bottlenecks(code):
        # 1. Fetch real aircraft data
        aircraft_data = fetch_real_adsb_data(code)
        
        # 2. Initialize model
        model = SimpleAirportBottleneckModel(SIMPLE_BOTTLENECK_CONFIG)
        
        # 3. Feed data into model
        airport_config = {'icao': code, 'coordinates': get_airport_coordinates(code)}
        analysis = model.predict_bottlenecks(aircraft_data, airport_config)
        
        # 4. Return predictions
        return jsonify(analysis)
    
    # Then in your JavaScript, you can:
    # - Fetch bottleneck data: fetch('/airport/JFK/bottlenecks')
    # - Display bottleneck locations on your map
    # - Show real-time predictions
    """)
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("✅ Feed aircraft data into model.predict_bottlenecks()")
    print("✅ Get predictions with exact coordinates")
    print("✅ Visualize bottleneck locations on your map")
    print("✅ Use real ADS-B data or sample scenarios")
    print("✅ Integrate with your existing Flask app")


if __name__ == "__main__":
    main()
