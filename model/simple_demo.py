"""
Simple Demo Script for MLP-based Airport Bottleneck Prediction

This script demonstrates how to use the simplified MLP model instead of 
the complex GNN-KAN hybrid architecture.
"""

import json
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simple_mlp_predictor import SimpleBottleneckPredictor, BottleneckPrediction
from simple_airport_model import SimpleAirportBottleneckModel
from simple_config import SIMPLE_BOTTLENECK_CONFIG


def create_sample_adsb_data():
    """Create sample ADS-B data for testing"""
    return {
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
                "lat": 40.6405,
                "lon": -73.7778,
                "alt_baro": 600,
                "track": 85,
                "gs": 140,
                "timestamp": datetime.now().isoformat()
            },
            {
                "flight": "SWA789",
                "t": "B737",
                "lat": 40.6398,
                "lon": -73.7795,
                "alt_baro": 400,
                "track": 95,
                "gs": 120,
                "timestamp": datetime.now().isoformat()
            },
            {
                "flight": "JBU012",
                "t": "A320",
                "lat": 40.6410,
                "lon": -73.7785,
                "alt_baro": 550,
                "track": 88,
                "gs": 135,
                "timestamp": datetime.now().isoformat()
            },
            {
                "flight": "AA345",
                "t": "B777",
                "lat": 40.6402,
                "lon": -73.7772,
                "alt_baro": 700,
                "track": 92,
                "gs": 160,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }


def create_airport_config():
    """Create sample airport configuration"""
    return {
        "icao": "KJFK",
        "name": "John F. Kennedy International Airport",
        "runways": ["04L/22R", "04R/22L", "09L/27R", "09R/27L", "13L/31R"],
        "gates": ["A1-A20", "B1-B20", "C1-C20", "D1-D20"],
        "coordinates": (40.63980103, -73.77890015)
    }


def main():
    """Main demo function"""
    print("🚀 Simple MLP Airport Bottleneck Prediction Demo")
    print("=" * 60)
    
    # Initialize the simplified model
    print("📊 Initializing simplified MLP model...")
    model = SimpleAirportBottleneckModel(SIMPLE_BOTTLENECK_CONFIG)
    
    # Create sample data
    print("✈️  Creating sample ADS-B data...")
    adsb_data = create_sample_adsb_data()
    airport_config = create_airport_config()
    
    print(f"📡 Monitoring {len(adsb_data['aircraft'])} aircraft at {airport_config['icao']}")
    
    # Run bottleneck prediction
    print("🔍 Running bottleneck prediction...")
    try:
        analysis = model.predict_bottlenecks(adsb_data, airport_config)
        
        # Display results
        print("\n📋 BOTTLENECK ANALYSIS RESULTS")
        print("=" * 40)
        
        print(f"🕐 Timestamp: {analysis['timestamp']}")
        print(f"🏢 Airport: {analysis['airport']}")
        print(f"✈️  Aircraft Monitored: {analysis['total_aircraft_monitored']}")
        print(f"📊 Model Type: {analysis['model_info']['model_type']}")
        
        # Airport summary
        summary = analysis['airport_summary']
        print(f"\n📈 AIRPORT SUMMARY:")
        print(f"   • Total Bottlenecks: {summary['total_bottlenecks_predicted']}")
        print(f"   • Highest Severity: {summary['highest_severity_level']}/5")
        print(f"   • Passengers at Risk: {summary['total_passengers_at_risk']}")
        print(f"   • Fuel Waste Estimate: {summary['total_fuel_waste_estimate']:.1f} gallons")
        print(f"   • Overall Risk Level: {summary['overall_delay_risk'].upper()}")
        
        # Individual bottleneck predictions
        if analysis['bottleneck_predictions']:
            print(f"\n🚨 BOTTLENECK PREDICTIONS:")
            for i, bottleneck in enumerate(analysis['bottleneck_predictions'], 1):
                print(f"\n   {i}. {bottleneck['type'].upper()}")
                print(f"      • Probability: {bottleneck['probability']:.2f}")
                print(f"      • Severity: {bottleneck['severity']}/5")
                print(f"      • Estimated Delay: {bottleneck['timing']['estimated_duration_minutes']:.1f} minutes")
                print(f"      • Passengers Affected: {bottleneck['impact_analysis']['passengers_affected']}")
                print(f"      • Fuel Waste: {bottleneck['impact_analysis']['fuel_waste_gallons']:.1f} gallons")
                print(f"      • Economic Impact: ${bottleneck['impact_analysis']['economic_impact_estimate']:.2f}")
                
                if bottleneck['recommended_mitigations']:
                    print(f"      • Recommended Action: {bottleneck['recommended_mitigations'][0]['action']}")
        else:
            print("\n✅ No bottlenecks predicted - operations running smoothly!")
        
        # Save results to file
        output_file = f"simple_bottleneck_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎯 Demo completed!")
    print("\nKey Benefits of Simplified MLP Model:")
    print("✅ Much simpler architecture (20 input features → 8 outputs)")
    print("✅ Faster execution (no complex graph operations)")
    print("✅ Easier to debug and understand")
    print("✅ Lower memory requirements")
    print("✅ Still provides accurate bottleneck predictions")


if __name__ == "__main__":
    main()
