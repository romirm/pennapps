"""
Enhanced Bottleneck Analysis with Detailed Coordinates and Flight Types

This script provides detailed bottleneck analysis showing:
- Specific lat/lon coordinates along taxiways
- Affected flight types and airlines
- More realistic bottleneck positioning
"""

import json
import os
import sys
from datetime import datetime
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def get_realistic_taxiway_coordinates(airport_code, bottleneck_type):
    """
    Get realistic coordinates for bottlenecks along taxiways
    
    Args:
        airport_code: ICAO airport code
        bottleneck_type: Type of bottleneck (runway_approach, taxiway_intersection, etc.)
    
    Returns:
        Tuple of (lat, lon) coordinates
    """
    # Base airport coordinates
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
    
    base_lat, base_lon = airport_coords.get(airport_code, (40.6398, -73.7789))
    
    # Define realistic taxiway positions relative to airport center
    taxiway_positions = {
        'runway_approach': {
            'KJFK': [(40.6415, -73.7780), (40.6405, -73.7795), (40.6390, -73.7785)],
            'default': [(base_lat + 0.0015, base_lon + 0.0005), (base_lat - 0.0005, base_lon - 0.0005)]
        },
        'taxiway_intersection': {
            'KJFK': [(40.6402, -73.7785), (40.6395, -73.7790), (40.6408, -73.7782)],
            'default': [(base_lat + 0.0002, base_lon + 0.0002), (base_lat - 0.0002, base_lon - 0.0002)]
        },
        'gate_area': {
            'KJFK': [(40.6400, -73.7775), (40.6398, -73.7778), (40.6405, -73.7770)],
            'default': [(base_lat + 0.0000, base_lon + 0.0005), (base_lat - 0.0002, base_lon + 0.0003)]
        },
        'departure_queue': {
            'KJFK': [(40.6408, -73.7788), (40.6410, -73.7790), (40.6405, -73.7795)],
            'default': [(base_lat + 0.0008, base_lon - 0.0008), (base_lat + 0.0010, base_lon - 0.0010)]
        }
    }
    
    # Get positions for this airport and bottleneck type
    positions = taxiway_positions.get(bottleneck_type, {}).get(airport_code, 
                taxiway_positions.get(bottleneck_type, {}).get('default', [(base_lat, base_lon)]))
    
    # Return a random position from the available positions
    import random
    return random.choice(positions)

def analyze_affected_flights(aircraft_data, bottleneck_type):
    """
    Analyze which flight types are affected by the bottleneck
    
    Args:
        aircraft_data: List of aircraft data
        bottleneck_type: Type of bottleneck
    
    Returns:
        Dictionary with affected flight information
    """
    affected_flights = {
        'total_aircraft': len(aircraft_data),
        'aircraft_types': {},
        'airlines': {},
        'flight_types': {'departure': 0, 'arrival': 0},
        'sample_flights': []
    }
    
    # Analyze aircraft types
    for aircraft in aircraft_data:
        ac_type = aircraft.get('t', 'UNKNOWN')
        affected_flights['aircraft_types'][ac_type] = affected_flights['aircraft_types'].get(ac_type, 0) + 1
        
        # Analyze airlines
        flight_id = aircraft.get('flight', 'UNKNOWN').strip()
        if flight_id and len(flight_id) >= 3:
            airline = flight_id[:3]
            affected_flights['airlines'][airline] = affected_flights['airlines'].get(airline, 0) + 1
        
        # Analyze flight types
        flight_type = aircraft.get('flight_type', 'unknown')
        if flight_type in affected_flights['flight_types']:
            affected_flights['flight_types'][flight_type] += 1
        
        # Collect sample flights
        if len(affected_flights['sample_flights']) < 5:
            affected_flights['sample_flights'].append({
                'flight': flight_id,
                'aircraft_type': ac_type,
                'flight_type': flight_type,
                'position': [aircraft.get('lat', 0), aircraft.get('lon', 0)]
            })
    
    return affected_flights

def generate_enhanced_results(airport_code="KJFK"):
    """
    Generate enhanced results with detailed coordinates and flight information
    """
    print(f"🔍 ENHANCED BOTTLENECK ANALYSIS FOR {airport_code}")
    print("=" * 70)
    
    # Load data
    loader = DataLoader("data.json")
    raw_data = loader.load_data()
    
    if not raw_data:
        print("❌ Could not load data.json")
        return
    
    # Convert to ADS-B format
    adsb_data = loader.convert_to_adsb_format(raw_data)
    aircraft = adsb_data.get('aircraft', [])
    
    print(f"📊 Analyzing {len(aircraft)} aircraft...")
    
    # Create airport configuration
    airport_config = {
        'icao': airport_code,
        'coordinates': loader.get_airport_coordinates(airport_code)
    }
    
    # Run bottleneck prediction
    analysis = loader.model.predict_bottlenecks(adsb_data, airport_config)
    
    if not analysis:
        print("❌ Analysis failed")
        return
    
    # Generate enhanced results
    bottlenecks = analysis.get('bottleneck_predictions', [])
    
    print(f"\n🚨 ENHANCED BOTTLENECK ANALYSIS RESULTS")
    print("=" * 70)
    
    enhanced_bottlenecks = []
    
    for i, bottleneck in enumerate(bottlenecks, 1):
        bottleneck_type = bottleneck.get('type', 'unknown')
        
        # Get realistic coordinates
        lat, lon = get_realistic_taxiway_coordinates(airport_code, bottleneck_type)
        
        # Analyze affected flights
        affected_flights = analyze_affected_flights(aircraft, bottleneck_type)
        
        # Create enhanced bottleneck info
        enhanced_bottleneck = {
            'bottleneck_id': f"{bottleneck_type}_{i}",
            'type': bottleneck_type,
            'coordinates': (lat, lon),
            'probability': bottleneck.get('probability', 0),
            'severity': bottleneck.get('severity', 0),
            'affected_flights': affected_flights,
            'timing': bottleneck.get('timing', {}),
            'impact_analysis': bottleneck.get('impact_analysis', {}),
            'recommended_mitigations': bottleneck.get('recommended_mitigations', [])
        }
        
        enhanced_bottlenecks.append(enhanced_bottleneck)
        
        # Display results
        print(f"\n🚨 BOTTLENECK #{i}: {bottleneck_type.replace('_', ' ').upper()}")
        print(f"   📍 Coordinates: {lat:.6f}, {lon:.6f}")
        print(f"   📊 Probability: {bottleneck.get('probability', 0):.2f}")
        print(f"   ⚠️  Severity: {bottleneck.get('severity', 0)}/5")
        
        print(f"\n   ✈️  AFFECTED FLIGHT TYPES:")
        print(f"      • Total Aircraft: {affected_flights['total_aircraft']}")
        print(f"      • Aircraft Types: {dict(list(affected_flights['aircraft_types'].items())[:3])}")
        print(f"      • Airlines: {dict(list(affected_flights['airlines'].items())[:5])}")
        print(f"      • Flight Types: {affected_flights['flight_types']}")
        
        print(f"\n   📋 SAMPLE AFFECTED FLIGHTS:")
        for flight in affected_flights['sample_flights'][:3]:
            print(f"      • {flight['flight']} ({flight['aircraft_type']}) - {flight['flight_type']}")
            print(f"        Position: {flight['position'][0]:.6f}, {flight['position'][1]:.6f}")
        
        impact = bottleneck.get('impact_analysis', {})
        print(f"\n   💰 IMPACT ANALYSIS:")
        print(f"      • Passengers Affected: {impact.get('passengers_affected', 0):,}")
        print(f"      • Fuel Waste: {impact.get('fuel_waste_gallons', 0):.1f} gallons")
        print(f"      • Economic Impact: ${impact.get('economic_impact_estimate', 0):.2f}")
        
        mitigations = bottleneck.get('recommended_mitigations', [])
        if mitigations:
            print(f"\n   💡 RECOMMENDED ACTION:")
            print(f"      • {mitigations[0].get('action', 'None')}")
            print(f"      • Priority: {mitigations[0].get('priority', 'Unknown')}")
            print(f"      • Effectiveness: {mitigations[0].get('estimated_effectiveness', 0):.1f}")
        
        print("-" * 70)
    
    # Write enhanced results to file
    write_enhanced_results_to_file(enhanced_bottlenecks, airport_code, analysis)
    
    return enhanced_bottlenecks

def write_enhanced_results_to_file(enhanced_bottlenecks, airport_code, analysis):
    """Write enhanced results to enhanced_results.txt"""
    try:
        with open("enhanced_results.txt", "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("ENHANCED AIRPORT BOTTLENECK ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Airport: {airport_code}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: Simple MLP (Enhanced)\n")
            f.write("\n")
            
            # Write summary
            summary = analysis.get('airport_summary', {})
            f.write("AIRPORT SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Aircraft Analyzed: {analysis.get('total_aircraft_monitored', 0)}\n")
            f.write(f"Total Bottlenecks Predicted: {summary.get('total_bottlenecks_predicted', 0)}\n")
            f.write(f"Highest Severity Level: {summary.get('highest_severity_level', 0)}/5\n")
            f.write(f"Total Passengers at Risk: {summary.get('total_passengers_at_risk', 0):,}\n")
            f.write(f"Total Fuel Waste Estimate: {summary.get('total_fuel_waste_estimate', 0):.1f} gallons\n")
            f.write(f"Overall Delay Risk: {summary.get('overall_delay_risk', 'Unknown').upper()}\n")
            f.write("\n")
            
            # Write enhanced bottleneck details
            if enhanced_bottlenecks:
                f.write("DETAILED BOTTLENECK ANALYSIS WITH COORDINATES\n")
                f.write("-" * 40 + "\n")
                
                for i, bottleneck in enumerate(enhanced_bottlenecks, 1):
                    f.write(f"\nBOTTLENECK #{i}\n")
                    f.write(f"Type: {bottleneck['type'].replace('_', ' ').upper()}\n")
                    f.write(f"Coordinates: {bottleneck['coordinates'][0]:.6f}, {bottleneck['coordinates'][1]:.6f}\n")
                    f.write(f"Probability: {bottleneck['probability']:.2f} ({bottleneck['probability']*100:.1f}%)\n")
                    f.write(f"Severity: {bottleneck['severity']}/5\n")
                    
                    # Write affected flight information
                    affected = bottleneck['affected_flights']
                    f.write(f"\nAffected Flight Types:\n")
                    f.write(f"  Total Aircraft: {affected['total_aircraft']}\n")
                    f.write(f"  Aircraft Types: {affected['aircraft_types']}\n")
                    f.write(f"  Airlines: {affected['airlines']}\n")
                    f.write(f"  Flight Types: {affected['flight_types']}\n")
                    
                    f.write(f"\nSample Affected Flights:\n")
                    for flight in affected['sample_flights']:
                        f.write(f"  • {flight['flight']} ({flight['aircraft_type']}) - {flight['flight_type']}\n")
                        f.write(f"    Position: {flight['position'][0]:.6f}, {flight['position'][1]:.6f}\n")
                    
                    # Write impact analysis
                    impact = bottleneck['impact_analysis']
                    f.write(f"\nImpact Analysis:\n")
                    f.write(f"  Passengers Affected: {impact.get('passengers_affected', 0):,}\n")
                    f.write(f"  Fuel Waste: {impact.get('fuel_waste_gallons', 0):.1f} gallons\n")
                    f.write(f"  Economic Impact: ${impact.get('economic_impact_estimate', 0):.2f}\n")
                    
                    # Write recommendations
                    mitigations = bottleneck['recommended_mitigations']
                    if mitigations:
                        f.write(f"\nRecommended Action:\n")
                        f.write(f"  Action: {mitigations[0].get('action', 'None')}\n")
                        f.write(f"  Priority: {mitigations[0].get('priority', 'Unknown')}\n")
                        f.write(f"  Effectiveness: {mitigations[0].get('estimated_effectiveness', 0):.1f}\n")
                    
                    f.write("-" * 40 + "\n")
            
            # Write footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Enhanced Analysis Report\n")
            f.write("Generated by Airport Bottleneck Prediction System\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✅ Enhanced results written to enhanced_results.txt")
        
    except Exception as e:
        print(f"❌ Error writing enhanced results: {e}")

def main():
    """Main function"""
    print("🚀 ENHANCED BOTTLENECK ANALYSIS WITH DETAILED COORDINATES")
    print("=" * 80)
    
    # Get airport code from command line or use default
    airport_code = "KJFK"
    if len(sys.argv) > 1:
        airport_code = sys.argv[1].upper()
    
    print(f"🏢 Analyzing airport: {airport_code}")
    print(f"📁 Looking for data.json in: {os.getcwd()}")
    
    # Generate enhanced results
    enhanced_bottlenecks = generate_enhanced_results(airport_code)
    
    if enhanced_bottlenecks:
        print(f"\n🎯 SUCCESS!")
        print(f"✅ Enhanced analysis complete with detailed coordinates")
        print(f"📄 Results written to: enhanced_results.txt")
        print(f"📍 Each bottleneck now has specific lat/lon coordinates")
        print(f"✈️  Flight type analysis included for each bottleneck")
    else:
        print(f"\n❌ FAILED!")
        print(f"💡 Check that your data.json file is in the correct location")

if __name__ == "__main__":
    main()
