"""
Example usage script for GNN-KAN Airport Bottleneck Prediction System

This script demonstrates how to use the bottleneck prediction system
with sample ADS-B data and airport configurations.
"""

import json
import torch
from datetime import datetime
from model import AirportBottleneckModel
from model.config import BOTTLENECK_CONFIG


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
            },
            {
                "flight": "FED345",
                "t": "B767F",
                "lat": 40.6390,
                "lon": -73.7760,
                "alt_baro": 0,
                "track": 0,
                "gs": 0,
                "timestamp": datetime.now().isoformat()
            }
        ]
    }


def create_sample_airport_config():
    """Create sample airport configuration"""
    return {
        "icao": "KJFK",
        "name": "John F. Kennedy International Airport",
        "city": "New York",
        "country": "USA",
        "runways": [
            {"id": "09L/27R", "length": 4423, "width": 45, "heading": 90},
            {"id": "09R/27L", "length": 4423, "width": 45, "heading": 90},
            {"id": "04L/22R", "length": 3682, "width": 45, "heading": 40},
            {"id": "04R/22L", "length": 2560, "width": 45, "heading": 40}
        ],
        "gates": ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2"],
        "taxiways": ["A", "B", "C", "D", "E", "F", "G", "H"]
    }


def run_bottleneck_analysis():
    """Run complete bottleneck analysis example"""
    print("🛫 GNN-KAN Airport Bottleneck Prediction System")
    print("=" * 50)
    
    # Initialize model
    print("Initializing model...")
    model = AirportBottleneckModel(BOTTLENECK_CONFIG)
    print("✅ Model initialized successfully")
    
    # Create sample data
    print("\nCreating sample data...")
    adsb_data = create_sample_adsb_data()
    airport_config = create_sample_airport_config()
    print(f"✅ Sample data created: {len(adsb_data['aircraft'])} aircraft")
    
    # Run prediction
    print("\nRunning bottleneck analysis...")
    try:
        analysis = model.predict_bottlenecks(adsb_data, airport_config)
        print("✅ Analysis completed successfully")
        
        # Display results
        print("\n📊 ANALYSIS RESULTS")
        print("=" * 30)
        
        # Airport summary
        summary = analysis['airport_summary']
        print(f"Airport: {analysis['airport']}")
        print(f"Aircraft monitored: {analysis['total_aircraft_monitored']}")
        print(f"Bottlenecks predicted: {summary['total_bottlenecks_predicted']}")
        print(f"Highest severity: {summary['highest_severity_level']}/5")
        print(f"Overall delay risk: {summary['overall_delay_risk'].upper()}")
        print(f"Passengers at risk: {summary['total_passengers_at_risk']}")
        print(f"Fuel waste estimate: {summary['total_fuel_waste_estimate']:.1f} gallons")
        
        # Bottleneck details
        if analysis['bottleneck_predictions']:
            print(f"\n🚨 BOTTLENECK DETAILS")
            print("-" * 30)
            
            for i, bottleneck in enumerate(analysis['bottleneck_predictions'], 1):
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
                
                if bottleneck['recommended_mitigations']:
                    print(f"  Recommended actions:")
                    for mitigation in bottleneck['recommended_mitigations'][:2]:  # Show top 2
                        print(f"    • {mitigation['action']} (effectiveness: {mitigation['estimated_effectiveness']:.1f})")
        else:
            print("\n✅ No bottlenecks detected - operations running smoothly!")
        
        # Save results
        output_file = f"bottleneck_analysis_{analysis['airport']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("This is expected for the initial implementation - the model needs training data")
        return False
    
    return True


def demonstrate_model_components():
    """Demonstrate individual model components"""
    print("\n🔧 MODEL COMPONENTS DEMONSTRATION")
    print("=" * 40)
    
    # Test ADS-B processor
    print("Testing ADS-B Data Processor...")
    from model.adsb_processor import ADSBDataProcessor
    
    processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
    adsb_data = create_sample_adsb_data()
    
    aircraft_list = processor.filter_airport_operations(adsb_data, "KJFK")
    print(f"✅ Processed {len(aircraft_list)} aircraft")
    
    bottleneck_zones = processor.identify_bottleneck_zones(aircraft_list)
    print(f"✅ Identified {len(bottleneck_zones)} bottleneck zones")
    
    # Test flight metadata processor
    print("\nTesting Flight Metadata Processor...")
    from model.flight_metadata import FlightMetadataProcessor
    
    metadata_processor = FlightMetadataProcessor()
    
    for aircraft in aircraft_list[:2]:  # Test first 2 aircraft
        flight_data = {
            'aircraft_type': aircraft.aircraft_type,
            'phase': aircraft.phase
        }
        passengers = metadata_processor.estimate_passenger_count(flight_data)
        fuel_consumption = metadata_processor.estimate_fuel_consumption(flight_data, 15)  # 15 min delay
        
        print(f"  {aircraft.flight_id} ({aircraft.aircraft_type}):")
        print(f"    Estimated passengers: {passengers}")
        print(f"    Fuel consumption (15min delay): {fuel_consumption['fuel_gallons']:.1f} gallons")
        print(f"    Fuel cost: ${fuel_consumption['fuel_cost_usd']:.2f}")
    
    print("✅ Flight metadata processing completed")
    
    # Test impact calculator
    print("\nTesting Impact Calculator...")
    from model.impact_calculator import ImpactCalculator
    
    impact_calculator = ImpactCalculator()
    
    # Create sample flight data
    sample_flights = [
        {
            'aircraft_type': 'B737',
            'phase': 'approach',
            'estimated_passengers': 150
        },
        {
            'aircraft_type': 'A320',
            'phase': 'ground',
            'estimated_passengers': 180
        }
    ]
    
    bottleneck_data = {
        'type': 'runway_approach_queue',
        'severity': 0.7,
        'delay_minutes': 15
    }
    
    economic_impact = impact_calculator.calculate_economic_impact(bottleneck_data, sample_flights)
    environmental_impact = impact_calculator.calculate_environmental_impact(bottleneck_data, sample_flights)
    passenger_impact = impact_calculator.calculate_passenger_impact(sample_flights, 15)
    
    print(f"  Economic impact: ${economic_impact['total_economic_impact_usd']:.2f}")
    print(f"  Environmental impact: {environmental_impact['co2_emissions_lbs']:.1f} lbs CO2")
    print(f"  Passenger impact: {passenger_impact['passengers_affected']} passengers affected")
    
    print("✅ Impact calculation completed")


def main():
    """Main function to run the example"""
    print("🚀 Starting GNN-KAN Airport Bottleneck Prediction System Example")
    print("=" * 60)
    
    # Check if PyTorch is available
    if not torch.cuda.is_available():
        print("⚠️  CUDA not available - using CPU (slower)")
    else:
        print("✅ CUDA available - using GPU acceleration")
    
    # Run main analysis
    success = run_bottleneck_analysis()
    
    if success:
        # Demonstrate components
        demonstrate_model_components()
        
        print("\n🎉 Example completed successfully!")
        print("\nNext steps:")
        print("1. Integrate with live ADS-B.lol API")
        print("2. Train the model on historical airport data")
        print("3. Implement real-time monitoring dashboard")
        print("4. Add weather integration for enhanced predictions")
    else:
        print("\n⚠️  Example completed with expected limitations")
        print("The model architecture is ready - training data integration needed")


if __name__ == "__main__":
    main()
