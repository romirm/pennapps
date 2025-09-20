"""
Test script for real-time bottleneck prediction

Usage:
    python model/test_bottleneck_prediction.py --flights UAL123 DAL456 SWA789 --airport KJFK
    python model/test_bottleneck_prediction.py --airport KJFK --radius 5
"""

import argparse
import json
import sys
from datetime import datetime
from model.flight_bottleneck_predictor import BottleneckPredictor


def test_specific_flights(flight_numbers: list, airport_icao: str):
    """Test bottleneck prediction for specific flights"""
    print(f"🛫 Testing bottleneck prediction for specific flights")
    print(f"Flights: {', '.join(flight_numbers)}")
    print(f"Airport: {airport_icao}")
    print("=" * 60)
    
    predictor = BottleneckPredictor()
    result = predictor.predict_bottlenecks_for_flights(flight_numbers, airport_icao)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'bottleneck_prediction_flights_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    return result


def test_airport_area(airport_icao: str, radius_nm: float):
    """Test bottleneck prediction for all aircraft near airport"""
    print(f"🛫 Testing bottleneck prediction for airport area")
    print(f"Airport: {airport_icao}")
    print(f"Radius: {radius_nm} nautical miles")
    print("=" * 60)
    
    predictor = BottleneckPredictor()
    result = predictor.predict_bottlenecks_near_airport(airport_icao, radius_nm)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'bottleneck_prediction_airport_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    return result


def create_sample_test_data():
    """Create sample test data for demonstration"""
    print("📋 Creating sample test data...")
    
    # Sample flight data (simulated)
    sample_adsb_data = {
        "aircraft": [
            {
                "flight": "UAL123",
                "t": "B737",
                "lat": 40.6413,
                "lon": -73.7781,
                "alt_baro": 500,
                "track": 90,
                "gs": 150,
                "timestamp": datetime.now().isoformat()
            },
            {
                "flight": "DAL456",
                "t": "A320",
                "lat": 40.6420,
                "lon": -73.7790,
                "alt_baro": 300,
                "track": 85,
                "gs": 120,
                "timestamp": datetime.now().isoformat()
            },
            {
                "flight": "SWA789",
                "t": "B737",
                "lat": 40.6400,
                "lon": -73.7770,
                "alt_baro": 0,
                "track": 0,
                "gs": 5,
                "timestamp": datetime.now().isoformat()
            },
            {
                "flight": "JBU012",
                "t": "A320",
                "lat": 40.6430,
                "lon": -73.7800,
                "alt_baro": 0,
                "track": 0,
                "gs": 0,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }
    
    # Save sample data
    with open('sample_adsb_data.json', 'w') as f:
        json.dump(sample_adsb_data, f, indent=2, default=str)
    
    print("✅ Sample data created: sample_adsb_data.json")
    return sample_adsb_data


def test_with_sample_data():
    """Test the model with sample data"""
    print("🧪 Testing model with sample data...")
    
    from model import AirportBottleneckModel
    from model.config import BOTTLENECK_CONFIG
    
    # Create sample data
    sample_data = create_sample_test_data()
    
    # Initialize model
    model = AirportBottleneckModel(BOTTLENECK_CONFIG)
    
    # Create airport config
    airport_config = {
        'icao': 'KJFK',
        'name': 'John F. Kennedy International Airport',
        'city': 'New York',
        'country': 'USA',
        'runways': [
            {'id': '09L/27R', 'length': 4423, 'width': 45, 'heading': 90},
            {'id': '09R/27L', 'length': 4423, 'width': 45, 'heading': 90}
        ],
        'gates': ['A1', 'A2', 'B1', 'B2'],
        'taxiways': ['A', 'B', 'C', 'D']
    }
    
    # Run prediction
    try:
        print("🔍 Running bottleneck analysis...")
        analysis = model.predict_bottlenecks(sample_data, airport_config)
        
        print("✅ Analysis completed successfully!")
        
        # Display results
        summary = analysis.get('airport_summary', {})
        print(f"\n📊 Results:")
        print(f"Aircraft monitored: {analysis.get('total_aircraft_monitored', 0)}")
        print(f"Bottlenecks predicted: {summary.get('total_bottlenecks_predicted', 0)}")
        print(f"Overall risk: {summary.get('overall_delay_risk', 'low').upper()}")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sample_bottleneck_analysis_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"💾 Results saved to: {filename}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("This is expected for the initial implementation - the model needs training data")
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Test bottleneck prediction system')
    parser.add_argument('--flights', nargs='+', help='Flight numbers to track (e.g., UAL123 DAL456)')
    parser.add_argument('--airport', help='Airport ICAO code (e.g., KJFK)')
    parser.add_argument('--radius', type=float, default=3.0, help='Search radius in nautical miles')
    parser.add_argument('--sample', action='store_true', help='Test with sample data')
    
    args = parser.parse_args()
    
    print("🚀 GNN-KAN Airport Bottleneck Prediction System - Test Script")
    print("=" * 70)
    
    if args.sample:
        # Test with sample data
        test_with_sample_data()
        
    elif args.flights and args.airport:
        # Test with specific flights
        test_specific_flights(args.flights, args.airport)
        
    elif args.airport:
        # Test with airport area
        test_airport_area(args.airport, args.radius)
        
    else:
        print("❌ Please provide either:")
        print("  --sample (test with sample data)")
        print("  --flights FLIGHT1 FLIGHT2 --airport ICAO (test specific flights)")
        print("  --airport ICAO --radius NM (test airport area)")
        print("\nExamples:")
        print("  python model/test_bottleneck_prediction.py --sample")
        print("  python model/test_bottleneck_prediction.py --flights UAL123 DAL456 --airport KJFK")
        print("  python model/test_bottleneck_prediction.py --airport KJFK --radius 5")
        
        # Run sample test as default
        print("\n🧪 Running sample test as default...")
        test_with_sample_data()


if __name__ == "__main__":
    main()
