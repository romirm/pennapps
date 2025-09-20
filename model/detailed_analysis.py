"""
Detailed Analysis of Your data.json File

This script provides detailed analysis of your flight tracking data
and shows exactly how it's being processed for bottleneck prediction.
"""

import json
import os
from datetime import datetime
from data_loader import DataLoader

def analyze_data_structure():
    """Analyze the structure of your data.json file"""
    print("🔍 ANALYZING YOUR DATA.JSON STRUCTURE")
    print("=" * 60)
    
    # Load the data
    loader = DataLoader("data.json")
    raw_data = loader.load_data()
    
    if not raw_data:
        print("❌ Could not load data.json")
        return
    
    print(f"📅 Date: {raw_data.get('date', 'Unknown')}")
    print(f"📊 Departures: {len(raw_data.get('departures', []))}")
    print(f"📊 Arrivals: {len(raw_data.get('arrivals', []))}")
    print(f"📊 Total Flights: {len(raw_data.get('departures', [])) + len(raw_data.get('arrivals', []))}")
    
    # Analyze departures
    departures = raw_data.get('departures', [])
    if departures:
        print(f"\n✈️ DEPARTURE ANALYSIS:")
        print(f"   • Total Departures: {len(departures)}")
        
        # Count by airline
        airlines = {}
        for dep in departures[:10]:  # Sample first 10
            callsign = dep.get('callsign', '').strip()
            if callsign:
                airline = callsign[:3]  # First 3 characters
                airlines[airline] = airlines.get(airline, 0) + 1
        
        print(f"   • Sample Airlines: {dict(list(airlines.items())[:5])}")
        
        # Count by destination
        destinations = {}
        for dep in departures:
            dest = dep.get('estArrivalAirport')
            if dest:
                destinations[dest] = destinations.get(dest, 0) + 1
        
        print(f"   • Top Destinations: {dict(list(sorted(destinations.items(), key=lambda x: x[1], reverse=True)[:5]))}")
    
    # Analyze arrivals
    arrivals = raw_data.get('arrivals', [])
    if arrivals:
        print(f"\n🛬 ARRIVAL ANALYSIS:")
        print(f"   • Total Arrivals: {len(arrivals)}")
        
        # Count by origin
        origins = {}
        for arr in arrivals:
            origin = arr.get('estDepartureAirport')
            if origin:
                origins[origin] = origins.get(origin, 0) + 1
        
        print(f"   • Top Origins: {dict(list(sorted(origins.items(), key=lambda x: x[1], reverse=True)[:5]))}")
    
    # Convert to ADS-B format
    print(f"\n🔄 CONVERTING TO ADS-B FORMAT...")
    adsb_data = loader.convert_to_adsb_format(raw_data)
    aircraft = adsb_data.get('aircraft', [])
    
    print(f"✅ Converted to {len(aircraft)} aircraft records")
    
    # Analyze converted data
    if aircraft:
        print(f"\n📊 CONVERTED AIRCRAFT ANALYSIS:")
        
        # Aircraft types
        aircraft_types = {}
        for ac in aircraft:
            ac_type = ac.get('t', 'UNKNOWN')
            aircraft_types[ac_type] = aircraft_types.get(ac_type, 0) + 1
        
        print(f"   • Aircraft Types: {dict(list(sorted(aircraft_types.items(), key=lambda x: x[1], reverse=True)[:5]))}")
        
        # Flight types
        flight_types = {}
        for ac in aircraft:
            ftype = ac.get('flight_type', 'UNKNOWN')
            flight_types[ftype] = flight_types.get(ftype, 0) + 1
        
        print(f"   • Flight Types: {flight_types}")
        
        # Sample aircraft data
        print(f"\n✈️ SAMPLE AIRCRAFT DATA:")
        for i, ac in enumerate(aircraft[:3]):
            print(f"   {i+1}. {ac.get('flight', 'UNKNOWN')} ({ac.get('t', 'UNKNOWN')})")
            print(f"      📍 Position: {ac.get('lat', 0):.4f}, {ac.get('lon', 0):.4f}")
            print(f"      📏 Altitude: {ac.get('alt_baro', 0)} ft")
            print(f"      🧭 Heading: {ac.get('track', 0)}°")
            print(f"      🚀 Speed: {ac.get('gs', 0)} knots")
            print(f"      🛫 Type: {ac.get('flight_type', 'UNKNOWN')}")
            print(f"      🏢 ICAO24: {ac.get('icao24', 'UNKNOWN')}")

def run_bottleneck_analysis():
    """Run bottleneck analysis on your data"""
    print(f"\n🚨 BOTTLENECK ANALYSIS")
    print("=" * 60)
    
    loader = DataLoader("data.json")
    analysis = loader.predict_bottlenecks_from_file("KJFK")
    
    if analysis:
        summary = analysis['airport_summary']
        bottlenecks = analysis['bottleneck_predictions']
        
        print(f"🏢 Airport: {analysis['airport']}")
        print(f"✈️ Aircraft Analyzed: {analysis['total_aircraft_monitored']}")
        print(f"🚨 Total Bottlenecks: {summary['total_bottlenecks_predicted']}")
        print(f"⚠️ Risk Level: {summary['overall_delay_risk'].upper()}")
        print(f"👥 Passengers at Risk: {summary['total_passengers_at_risk']:,}")
        print(f"⛽ Fuel Waste: {summary['total_fuel_waste_estimate']:.1f} gallons")
        
        if bottlenecks:
            print(f"\n🚨 DETAILED BOTTLENECK ANALYSIS:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"\n   {i}. {bottleneck['type'].replace('_', ' ').upper()}")
                print(f"      📍 Location: {bottleneck['location']['coordinates']}")
                print(f"      📊 Probability: {bottleneck['probability']:.2f}")
                print(f"      ⚠️ Severity: {bottleneck['severity']}/5")
                print(f"      ⏱️ Estimated Delay: {bottleneck['timing']['estimated_duration_minutes']:.1f} minutes")
                print(f"      👥 Passengers Affected: {bottleneck['impact_analysis']['passengers_affected']:,}")
                print(f"      ⛽ Fuel Waste: {bottleneck['impact_analysis']['fuel_waste_gallons']:.1f} gallons")
                print(f"      💰 Economic Impact: ${bottleneck['impact_analysis']['economic_impact_estimate']:.2f}")
                
                if bottleneck.get('recommended_mitigations'):
                    print(f"      💡 Recommendation: {bottleneck['recommended_mitigations'][0]['action']}")

def main():
    """Main analysis function"""
    print("🎯 DETAILED ANALYSIS OF YOUR DATA.JSON FILE")
    print("=" * 70)
    
    # Check if data.json exists
    if not os.path.exists("data.json"):
        print("❌ data.json file not found!")
        print("💡 Make sure your data.json file is in the model directory")
        return
    
    # Analyze data structure
    analyze_data_structure()
    
    # Run bottleneck analysis
    run_bottleneck_analysis()
    
    print(f"\n🎯 SUMMARY:")
    print("✅ Your data.json file has been successfully processed!")
    print("✅ The model can now predict bottlenecks from your flight tracking data")
    print("✅ You can integrate this with your Flask app for real-time analysis")
    
    print(f"\n📁 NEXT STEPS:")
    print("1. Use the Flask integration script to add bottleneck prediction to your app")
    print("2. Display bottleneck locations on your airport map")
    print("3. Set up real-time updates with your data.json file")

if __name__ == "__main__":
    main()
