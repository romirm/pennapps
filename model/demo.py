"""
Demo script for GNN-KAN Airport Bottleneck Prediction System

This script demonstrates how to use the system with real flight data.
"""

import json
from datetime import datetime
from model.flight_bottleneck_predictor import BottleneckPredictor


def demo_with_real_flights():
    """Demo with real flight numbers"""
    print("🚀 GNN-KAN Airport Bottleneck Prediction System - Demo")
    print("=" * 60)
    
    # Initialize predictor
    predictor = BottleneckPredictor()
    
    # Example 1: Test with specific flights
    print("\n📋 Example 1: Analyzing specific flights")
    print("-" * 40)
    
    # You can replace these with real active flight numbers
    flight_numbers = ['UAL123', 'DAL456', 'SWA789']
    airport = 'KJFK'
    
    print(f"Tracking flights: {', '.join(flight_numbers)} at {airport}")
    print("Note: These are example flights - replace with real active flights")
    
    result = predictor.predict_bottlenecks_for_flights(flight_numbers, airport)
    
    # Example 2: Test with airport area
    print(f"\n📋 Example 2: Analyzing all aircraft near {airport}")
    print("-" * 40)
    
    result2 = predictor.predict_bottlenecks_near_airport(airport, radius_nm=3.0)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'demo_results_flights_{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    with open(f'demo_results_airport_{timestamp}.json', 'w') as f:
        json.dump(result2, f, indent=2, default=str)
    
    print(f"\n💾 Demo results saved with timestamp: {timestamp}")
    
    return result, result2


def demo_with_sample_data():
    """Demo with sample data"""
    print("🧪 Demo with Sample Data")
    print("=" * 30)
    
    from model import AirportBottleneckModel
    from model.config import BOTTLENECK_CONFIG
    
    # Create sample data
    sample_data = {
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
    
    airport_config = {
        'icao': 'KJFK',
        'name': 'John F. Kennedy International Airport',
        'city': 'New York',
        'country': 'USA',
        'runways': [
            {'id': '09L/27R', 'length': 4423, 'width': 45, 'heading': 90},
            {'id': '09R/27L', 'length': 4423, 'width': 45, 'heading': 90},
            {'id': '04L/22R', 'length': 3682, 'width': 45, 'heading': 40},
            {'id': '04R/22L', 'length': 2560, 'width': 45, 'heading': 40}
        ],
        'gates': ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2'],
        'taxiways': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    }
    
    model = AirportBottleneckModel(BOTTLENECK_CONFIG)
    
    try:
        print("🔍 Running sample analysis...")
        analysis = model.predict_bottlenecks(sample_data, airport_config)
        
        print("✅ Sample analysis completed!")
        
        # Display detailed results
        print(f"\n📊 DETAILED RESULTS")
        print("=" * 30)
        
        summary = analysis.get('airport_summary', {})
        print(f"Airport: {analysis.get('airport', 'UNKNOWN')}")
        print(f"Aircraft monitored: {analysis.get('total_aircraft_monitored', 0)}")
        print(f"Bottlenecks predicted: {summary.get('total_bottlenecks_predicted', 0)}")
        print(f"Highest severity: {summary.get('highest_severity_level', 1)}/5")
        print(f"Overall delay risk: {summary.get('overall_delay_risk', 'low').upper()}")
        print(f"Passengers at risk: {summary.get('total_passengers_at_risk', 0)}")
        print(f"Fuel waste estimate: {summary.get('total_fuel_waste_estimate', 0):.1f} gallons")
        
        # Bottleneck details
        bottlenecks = analysis.get('bottleneck_predictions', [])
        if bottlenecks:
            print(f"\n🚨 BOTTLENECK DETAILS")
            print("-" * 30)
            
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"\nBottleneck #{i}: {bottleneck['type'].replace('_', ' ').title()}")
                print(f"  Location: {bottleneck['location']['zone']}")
                print(f"  Probability: {bottleneck['probability']:.2f}")
                print(f"  Severity: {bottleneck['severity']}/5")
                print(f"  Duration: {bottleneck['timing']['estimated_duration_minutes']:.1f} minutes")
                print(f"  Aircraft affected: {len(bottleneck['aircraft_affected'])}")
                
                impact = bottleneck['impact_analysis']
                print(f"  Passengers affected: {impact['passengers_affected']}")
                print(f"  Fuel waste: {impact['fuel_waste_gallons']:.1f} gallons")
                print(f"  Fuel cost: ${impact['fuel_cost_estimate']:.2f}")
                print(f"  CO2 emissions: {impact['co2_emissions_lbs']:.1f} lbs")
                print(f"  Economic impact: ${impact['economic_impact_estimate']:.2f}")
                
                if bottleneck.get('recommended_mitigations'):
                    print(f"  Recommended actions:")
                    for mitigation in bottleneck['recommended_mitigations'][:2]:
                        print(f"    • {mitigation['action']} (effectiveness: {mitigation['estimated_effectiveness']:.1f})")
        else:
            print("\n✅ No bottlenecks detected - operations running smoothly!")
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sample_demo_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error during sample analysis: {e}")
        print("This is expected for the initial implementation - the model needs training data")
        return None


def main():
    """Main demo function"""
    print("🎯 Choose demo type:")
    print("1. Demo with real flight data (requires active flights)")
    print("2. Demo with sample data (works offline)")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == '1':
        demo_with_real_flights()
    elif choice == '2':
        demo_with_sample_data()
    else:
        print("❌ Invalid choice. Running sample demo...")
        demo_with_sample_data()
    
    print(f"\n🎉 Demo completed!")
    print(f"\n📋 Next steps:")
    print("1. Replace example flight numbers with real active flights")
    print("2. Use actual airport ICAO codes (KJFK, KLAX, KPHL, etc.)")
    print("3. The system will fetch real-time data from ADS-B.lol")
    print("4. Bottleneck predictions will be generated based on current aircraft positions")
    print("5. Results are saved as JSON files for further analysis")


if __name__ == "__main__":
    main()
