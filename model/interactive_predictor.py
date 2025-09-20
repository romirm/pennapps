"""
Interactive Bottleneck Prediction Tool

A simple interactive tool to test bottleneck prediction with real flight data.
"""

import json
from datetime import datetime
from model.flight_bottleneck_predictor import BottleneckPredictor


def interactive_bottleneck_prediction():
    """Interactive bottleneck prediction tool"""
    print("🚀 Interactive Bottleneck Prediction Tool")
    print("=" * 50)
    
    predictor = BottleneckPredictor()
    
    while True:
        print("\n📋 Choose an option:")
        print("1. Predict bottlenecks for specific flights")
        print("2. Predict bottlenecks for all aircraft near airport")
        print("3. Test with sample data")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Specific flights
            print("\n🛫 Specific Flight Analysis")
            print("-" * 30)
            
            flights_input = input("Enter flight numbers (comma-separated): ").strip()
            if not flights_input:
                print("❌ No flight numbers provided")
                continue
                
            flights = [f.strip().upper() for f in flights_input.split(',')]
            
            airport = input("Enter airport ICAO code (e.g., KJFK): ").strip().upper()
            if not airport:
                print("❌ No airport code provided")
                continue
            
            print(f"\n🔍 Analyzing flights: {', '.join(flights)} at {airport}")
            result = predictor.predict_bottlenecks_for_flights(flights, airport)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'bottleneck_flights_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 Results saved to: {filename}")
            
        elif choice == '2':
            # Airport area
            print("\n🛫 Airport Area Analysis")
            print("-" * 30)
            
            airport = input("Enter airport ICAO code (e.g., KJFK): ").strip().upper()
            if not airport:
                print("❌ No airport code provided")
                continue
            
            radius_input = input("Enter search radius in nautical miles (default 3): ").strip()
            try:
                radius = float(radius_input) if radius_input else 3.0
            except ValueError:
                radius = 3.0
                print("⚠️ Invalid radius, using default 3.0 nm")
            
            print(f"\n🔍 Analyzing aircraft near {airport} within {radius} nm")
            result = predictor.predict_bottlenecks_near_airport(airport, radius)
            
            # Save results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'bottleneck_airport_{timestamp}.json'
            with open(filename, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"💾 Results saved to: {filename}")
            
        elif choice == '3':
            # Sample data test
            print("\n🧪 Sample Data Test")
            print("-" * 30)
            
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
                    {'id': '09R/27L', 'length': 4423, 'width': 45, 'heading': 90}
                ],
                'gates': ['A1', 'A2', 'B1', 'B2'],
                'taxiways': ['A', 'B', 'C', 'D']
            }
            
            model = AirportBottleneckModel(BOTTLENECK_CONFIG)
            
            try:
                print("🔍 Running sample analysis...")
                analysis = model.predict_bottlenecks(sample_data, airport_config)
                
                print("✅ Sample analysis completed!")
                
                # Display summary
                summary = analysis.get('airport_summary', {})
                print(f"\n📊 Sample Results:")
                print(f"Aircraft monitored: {analysis.get('total_aircraft_monitored', 0)}")
                print(f"Bottlenecks predicted: {summary.get('total_bottlenecks_predicted', 0)}")
                print(f"Overall risk: {summary.get('overall_delay_risk', 'low').upper()}")
                
                # Save results
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'sample_analysis_{timestamp}.json'
                with open(filename, 'w') as f:
                    json.dump(analysis, f, indent=2, default=str)
                print(f"💾 Results saved to: {filename}")
                
            except Exception as e:
                print(f"❌ Error during sample analysis: {e}")
                print("This is expected for the initial implementation")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1-4.")
        
        # Ask if user wants to continue
        continue_choice = input("\nWould you like to run another analysis? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("👋 Goodbye!")
            break


def quick_test():
    """Quick test function for demonstration"""
    print("🚀 Quick Bottleneck Prediction Test")
    print("=" * 40)
    
    # Test with sample data
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
            {'id': '09R/27L', 'length': 4423, 'width': 45, 'heading': 90}
        ],
        'gates': ['A1', 'A2', 'B1', 'B2'],
        'taxiways': ['A', 'B', 'C', 'D']
    }
    
    model = AirportBottleneckModel(BOTTLENECK_CONFIG)
    
    try:
        print("🔍 Running quick test...")
        analysis = model.predict_bottlenecks(sample_data, airport_config)
        
        print("✅ Quick test completed!")
        
        # Display results
        summary = analysis.get('airport_summary', {})
        print(f"\n📊 Results:")
        print(f"Aircraft monitored: {analysis.get('total_aircraft_monitored', 0)}")
        print(f"Bottlenecks predicted: {summary.get('total_bottlenecks_predicted', 0)}")
        print(f"Overall risk: {summary.get('overall_delay_risk', 'low').upper()}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error during quick test: {e}")
        print("This is expected for the initial implementation")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        quick_test()
    else:
        interactive_bottleneck_prediction()
