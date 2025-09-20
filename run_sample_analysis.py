"""
Sample Data Bottleneck Prediction Script

Loads sample aircraft data and runs it through the GNN-KAN model
to generate bottleneck predictions and impact analysis.
"""

import json
import sys
import os
from datetime import datetime

# Add the model directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'model'))

try:
    from model import AirportBottleneckModel
    from model.config import BOTTLENECK_CONFIG
    from model.adsb_processor import ADSBDataProcessor
    from model.flight_metadata import FlightMetadataProcessor
    from model.impact_calculator import ImpactCalculator
except ImportError as e:
    print(f"❌ Error importing model components: {e}")
    print("Make sure you're in the correct directory and the model files exist")
    sys.exit(1)


def load_sample_data():
    """Load sample aircraft data from JSON file"""
    try:
        with open('sample_aircraft_data.json', 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded sample data: {len(data.get('aircraft', []))} aircraft")
        return data
    except FileNotFoundError:
        print("❌ sample_aircraft_data.json not found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None


def create_airport_config():
    """Create JFK airport configuration"""
    return {
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
        'gates': ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'D1', 'D2'],
        'taxiways': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K']
    }


def run_bottleneck_analysis():
    """Run complete bottleneck analysis on sample data"""
    print("🚀 GNN-KAN Airport Bottleneck Prediction System")
    print("Sample Data Analysis")
    print("=" * 50)
    
    # Load sample data
    print("\n📡 Loading sample aircraft data...")
    adsb_data = load_sample_data()
    if not adsb_data:
        return None
    
    # Display aircraft information
    print(f"\n📋 Aircraft Information:")
    print("-" * 30)
    for i, aircraft in enumerate(adsb_data['aircraft'], 1):
        flight_id = aircraft.get('flight', 'UNKNOWN')
        aircraft_type = aircraft.get('t', 'UNKNOWN')
        lat = aircraft.get('lat', 0)
        lon = aircraft.get('lon', 0)
        alt = aircraft.get('alt_baro', 0)
        speed = aircraft.get('gs', 0)
        heading = aircraft.get('track', 0)
        
        print(f"{i}. {flight_id} ({aircraft_type})")
        print(f"   Position: {lat:.4f}, {lon:.4f}")
        print(f"   Altitude: {alt} ft, Speed: {speed} kts, Heading: {heading}°")
    
    # Create airport configuration
    print(f"\n🏢 Airport Configuration:")
    airport_config = create_airport_config()
    print(f"Airport: {airport_config['name']} ({airport_config['icao']})")
    print(f"Runways: {len(airport_config['runways'])}")
    print(f"Gates: {len(airport_config['gates'])}")
    print(f"Taxiways: {len(airport_config['taxiways'])}")
    
    # Initialize model
    print(f"\n🧠 Initializing GNN-KAN Model...")
    try:
        model = AirportBottleneckModel(BOTTLENECK_CONFIG)
        print("✅ Model initialized successfully")
    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return None
    
    # Run prediction
    print(f"\n🔍 Running Bottleneck Analysis...")
    try:
        analysis = model.predict_bottlenecks(adsb_data, airport_config)
        print("✅ Analysis completed successfully!")
        
        # Display results
        display_analysis_results(analysis)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'bottleneck_analysis_results_{timestamp}.json'
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        print(f"\n💾 Results saved to: {filename}")
        
        return analysis
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("This is expected for the initial implementation - the model needs training data")
        
        # Try a simplified analysis
        print("\n🔄 Running simplified analysis...")
        return run_simplified_analysis(adsb_data, airport_config)


def run_simplified_analysis(adsb_data, airport_config):
    """Run a simplified analysis when the full model fails"""
    print("📊 Simplified Bottleneck Analysis")
    print("-" * 40)
    
    # Initialize components
    adsb_processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
    metadata_processor = FlightMetadataProcessor()
    impact_calculator = ImpactCalculator()
    
    # Process aircraft data
    aircraft_list = adsb_processor.filter_airport_operations(adsb_data, 'KJFK')
    print(f"✅ Processed {len(aircraft_list)} aircraft")
    
    # Identify bottleneck zones
    bottleneck_zones = adsb_processor.identify_bottleneck_zones(aircraft_list)
    print(f"✅ Identified {len(bottleneck_zones)} bottleneck zones")
    
    # Calculate impacts
    total_passengers = 0
    total_fuel_waste = 0
    total_economic_impact = 0
    
    for aircraft in aircraft_list:
        flight_data = {
            'aircraft_type': aircraft.aircraft_type,
            'phase': aircraft.phase
        }
        
        passengers = metadata_processor.estimate_passenger_count(flight_data)
        fuel_consumption = metadata_processor.estimate_fuel_consumption(flight_data, 15)  # 15 min delay
        
        total_passengers += passengers
        total_fuel_waste += fuel_consumption['fuel_gallons']
        total_economic_impact += fuel_consumption['fuel_cost_usd'] + (passengers * 15 * 2.50)  # 15 min delay
    
    # Create simplified results
    simplified_results = {
        'timestamp': datetime.now().isoformat(),
        'airport': 'KJFK',
        'analysis_type': 'simplified',
        'total_aircraft_monitored': len(aircraft_list),
        'bottleneck_zones_identified': len(bottleneck_zones),
        'impact_summary': {
            'total_passengers_at_risk': total_passengers,
            'total_fuel_waste_gallons': total_fuel_waste,
            'total_economic_impact_usd': total_economic_impact,
            'average_delay_minutes': 15
        },
        'aircraft_details': []
    }
    
    # Add aircraft details
    for aircraft in aircraft_list:
        flight_data = {
            'aircraft_type': aircraft.aircraft_type,
            'phase': aircraft.phase
        }
        passengers = metadata_processor.estimate_passenger_count(flight_data)
        
        simplified_results['aircraft_details'].append({
            'flight_id': aircraft.flight_id,
            'aircraft_type': aircraft.aircraft_type,
            'phase': aircraft.phase,
            'estimated_passengers': passengers,
            'position': [aircraft.latitude, aircraft.longitude],
            'altitude': aircraft.altitude,
            'speed': aircraft.speed
        })
    
    # Display simplified results
    print(f"\n📊 SIMPLIFIED ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Airport: {simplified_results['airport']}")
    print(f"Aircraft monitored: {simplified_results['total_aircraft_monitored']}")
    print(f"Bottleneck zones: {simplified_results['bottleneck_zones_identified']}")
    print(f"Passengers at risk: {simplified_results['impact_summary']['total_passengers_at_risk']}")
    print(f"Fuel waste estimate: {simplified_results['impact_summary']['total_fuel_waste_gallons']:.1f} gallons")
    print(f"Economic impact: ${simplified_results['impact_summary']['total_economic_impact_usd']:.2f}")
    
    # Save simplified results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'simplified_analysis_results_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(simplified_results, f, indent=2, default=str)
    
    print(f"\n💾 Simplified results saved to: {filename}")
    
    return simplified_results


def display_analysis_results(analysis):
    """Display detailed analysis results"""
    print(f"\n📊 DETAILED ANALYSIS RESULTS")
    print("=" * 50)
    
    # Airport summary
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


def main():
    """Main function"""
    print("🎯 Sample Data Bottleneck Prediction")
    print("This script will load sample aircraft data and run bottleneck analysis")
    print("=" * 60)
    
    # Run the analysis
    results = run_bottleneck_analysis()
    
    if results:
        print(f"\n🎉 Analysis completed successfully!")
        print(f"\n📋 Summary:")
        if results.get('analysis_type') == 'simplified':
            print(f"- Simplified analysis completed")
            print(f"- {results['total_aircraft_monitored']} aircraft analyzed")
            print(f"- {results['bottleneck_zones_identified']} bottleneck zones identified")
            print(f"- ${results['impact_summary']['total_economic_impact_usd']:.2f} total economic impact")
        else:
            print(f"- Full GNN-KAN analysis completed")
            print(f"- {results.get('total_aircraft_monitored', 0)} aircraft analyzed")
            print(f"- {results.get('airport_summary', {}).get('total_bottlenecks_predicted', 0)} bottlenecks predicted")
    else:
        print(f"\n❌ Analysis failed")
    
    print(f"\n📁 Files created:")
    print(f"- sample_aircraft_data.json (input data)")
    print(f"- bottleneck_analysis_results_*.json (output results)")


if __name__ == "__main__":
    main()
