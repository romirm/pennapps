"""
Time-Based Bottleneck Analysis

This script analyzes bottlenecks based on:
- Time ranges when bottlenecks occur (when 2+ aircraft are affected)
- Specific flight numbers affected during each time period
- Geographic locations of bottlenecks
- Duration and timing of congestion events
"""

import json
import os
import sys
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def parse_timestamp(timestamp_str):
    """Parse timestamp string to datetime object"""
    try:
        if isinstance(timestamp_str, (int, float)):
            return datetime.fromtimestamp(timestamp_str)
        elif isinstance(timestamp_str, str):
            # Try different timestamp formats
            for fmt in ['%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%d %H:%M:%S']:
                try:
                    return datetime.strptime(timestamp_str.replace('Z', ''), fmt)
                except ValueError:
                    continue
        return datetime.now()
    except:
        return datetime.now()

def group_aircraft_by_time(aircraft_data, time_window_minutes=15):
    """
    Group aircraft by time windows to detect congestion periods
    
    Args:
        aircraft_data: List of aircraft data
        time_window_minutes: Time window size in minutes
    
    Returns:
        Dictionary with time windows as keys and aircraft lists as values
    """
    time_groups = defaultdict(list)
    
    for aircraft in aircraft_data:
        timestamp = aircraft.get('timestamp', '')
        dt = parse_timestamp(timestamp)
        
        # Round to nearest time window
        window_start = dt.replace(minute=(dt.minute // time_window_minutes) * time_window_minutes, second=0, microsecond=0)
        
        time_groups[window_start].append(aircraft)
    
    return dict(time_groups)

def detect_bottlenecks_by_time(aircraft_data, airport_code, min_aircraft_threshold=2):
    """
    Detect bottlenecks based on time periods with high aircraft density
    
    Args:
        aircraft_data: List of aircraft data
        airport_code: ICAO airport code
        min_aircraft_threshold: Minimum aircraft count to consider a bottleneck
    
    Returns:
        List of bottleneck events with timing and affected flights
    """
    # Group aircraft by 15-minute time windows
    time_groups = group_aircraft_by_time(aircraft_data, time_window_minutes=15)
    
    bottlenecks = []
    
    for time_window, aircraft_in_window in time_groups.items():
        if len(aircraft_in_window) >= min_aircraft_threshold:
            # This time window has enough aircraft to be considered a bottleneck
            
            # Analyze aircraft types and airlines
            aircraft_types = {}
            airlines = {}
            flight_numbers = []
            
            for aircraft in aircraft_in_window:
                ac_type = aircraft.get('t', 'UNKNOWN')
                aircraft_types[ac_type] = aircraft_types.get(ac_type, 0) + 1
                
                flight_id = aircraft.get('flight', 'UNKNOWN').strip()
                if flight_id and len(flight_id) >= 3:
                    airline = flight_id[:3]
                    airlines[airline] = airlines.get(airline, 0) + 1
                
                flight_numbers.append(flight_id)
            
            # Determine bottleneck type based on aircraft distribution
            bottleneck_type = determine_bottleneck_type(aircraft_in_window, aircraft_types)
            
            # Calculate time range
            timestamps = [parse_timestamp(ac.get('timestamp', '')) for ac in aircraft_in_window]
            start_time = min(timestamps)
            end_time = max(timestamps)
            duration_minutes = (end_time - start_time).total_seconds() / 60
            
            # Get representative coordinates (average of aircraft positions)
            avg_lat = np.mean([ac.get('lat', 0) for ac in aircraft_in_window])
            avg_lon = np.mean([ac.get('lon', 0) for ac in aircraft_in_window])
            
            bottleneck = {
                'time_window': time_window,
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': duration_minutes,
                'aircraft_count': len(aircraft_in_window),
                'bottleneck_type': bottleneck_type,
                'coordinates': (avg_lat, avg_lon),
                'aircraft_types': aircraft_types,
                'airlines': airlines,
                'affected_flights': flight_numbers,
                'sample_aircraft': aircraft_in_window[:5]  # First 5 aircraft as samples
            }
            
            bottlenecks.append(bottleneck)
    
    # Sort by time window
    bottlenecks.sort(key=lambda x: x['time_window'])
    
    return bottlenecks

def determine_bottleneck_type(aircraft_list, aircraft_types):
    """
    Determine the type of bottleneck based on aircraft characteristics
    
    Args:
        aircraft_list: List of aircraft in the time window
        aircraft_types: Dictionary of aircraft type counts
    
    Returns:
        String describing the bottleneck type
    """
    total_aircraft = len(aircraft_list)
    
    # Analyze flight types
    departure_count = sum(1 for ac in aircraft_list if ac.get('flight_type') == 'departure')
    arrival_count = sum(1 for ac in aircraft_list if ac.get('flight_type') == 'arrival')
    
    # Analyze aircraft types
    b737_count = aircraft_types.get('B737', 0)
    a320_count = aircraft_types.get('A320', 0)
    widebody_count = sum(count for ac_type, count in aircraft_types.items() 
                        if ac_type in ['B777', 'B787', 'A330', 'A350'])
    
    # Determine bottleneck type
    if departure_count > arrival_count * 1.5:
        return "DEPARTURE_CONGESTION"
    elif arrival_count > departure_count * 1.5:
        return "ARRIVAL_CONGESTION"
    elif b737_count > total_aircraft * 0.7:
        return "NARROW_BODY_CONGESTION"
    elif widebody_count > total_aircraft * 0.3:
        return "WIDEBODY_CONGESTION"
    else:
        return "MIXED_TRAFFIC_CONGESTION"

def generate_time_based_analysis(airport_code="KJFK"):
    """
    Generate time-based bottleneck analysis
    """
    print(f"🕐 TIME-BASED BOTTLENECK ANALYSIS FOR {airport_code}")
    print("=" * 80)
    
    # Load data
    loader = DataLoader("data.json")
    raw_data = loader.load_data()
    
    if not raw_data:
        print("❌ Could not load data.json")
        return
    
    # Convert to ADS-B format
    adsb_data = loader.convert_to_adsb_format(raw_data)
    aircraft = adsb_data.get('aircraft', [])
    
    print(f"📊 Analyzing {len(aircraft)} aircraft over time...")
    
    # Detect bottlenecks by time
    bottlenecks = detect_bottlenecks_by_time(aircraft, airport_code, min_aircraft_threshold=2)
    
    if not bottlenecks:
        print("✅ No bottlenecks detected - all time periods have < 2 aircraft")
        return
    
    print(f"\n🚨 TIME-BASED BOTTLENECK DETECTION RESULTS")
    print("=" * 80)
    print(f"📅 Found {len(bottlenecks)} bottleneck time periods")
    print(f"⏰ Analysis period: {min(b['time_window'] for b in bottlenecks)} to {max(b['time_window'] for b in bottlenecks)}")
    
    # Display each bottleneck
    for i, bottleneck in enumerate(bottlenecks, 1):
        print(f"\n🚨 BOTTLENECK #{i}: {bottleneck['bottleneck_type'].replace('_', ' ').title()}")
        print(f"   ⏰ Time Window: {bottleneck['time_window'].strftime('%Y-%m-%d %H:%M')}")
        print(f"   📅 Duration: {bottleneck['duration_minutes']:.1f} minutes")
        print(f"   📍 Location: {bottleneck['coordinates'][0]:.6f}, {bottleneck['coordinates'][1]:.6f}")
        print(f"   ✈️  Aircraft Count: {bottleneck['aircraft_count']}")
        
        print(f"\n   🛩️  AIRCRAFT TYPES:")
        for ac_type, count in sorted(bottleneck['aircraft_types'].items(), key=lambda x: x[1], reverse=True):
            print(f"      • {ac_type}: {count} aircraft")
        
        print(f"\n   🏢 AIRLINES:")
        for airline, count in sorted(bottleneck['airlines'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"      • {airline}: {count} flights")
        
        print(f"\n   📋 AFFECTED FLIGHT NUMBERS:")
        unique_flights = list(set(bottleneck['affected_flights']))[:15]  # Show first 15 unique flights
        for flight in unique_flights:
            if flight != 'UNKNOWN':
                print(f"      • {flight}")
        
        if len(unique_flights) > 15:
            print(f"      ... and {len(unique_flights) - 15} more flights")
        
        print(f"\n   📊 SAMPLE AIRCRAFT POSITIONS:")
        for ac in bottleneck['sample_aircraft'][:3]:
            flight_id = ac.get('flight', 'UNKNOWN')
            ac_type = ac.get('t', 'UNKNOWN')
            lat = ac.get('lat', 0)
            lon = ac.get('lon', 0)
            print(f"      • {flight_id} ({ac_type}): {lat:.6f}, {lon:.6f}")
        
        print("-" * 80)
    
    # Write time-based results to file
    write_time_based_results_to_file(bottlenecks, airport_code)
    
    return bottlenecks

def write_time_based_results_to_file(bottlenecks, airport_code):
    """Write time-based bottleneck results to time_based_results.txt"""
    try:
        with open("time_based_results.txt", "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("TIME-BASED BOTTLENECK ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Airport: {airport_code}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Type: Time-Based Bottleneck Detection\n")
            f.write(f"Minimum Aircraft Threshold: 2 aircraft per time window\n")
            f.write(f"Time Window Size: 15 minutes\n")
            f.write("\n")
            
            # Write summary
            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Bottleneck Periods: {len(bottlenecks)}\n")
            if bottlenecks:
                f.write(f"Analysis Period: {min(b['time_window'] for b in bottlenecks)} to {max(b['time_window'] for b in bottlenecks)}\n")
                f.write(f"Total Aircraft Affected: {sum(b['aircraft_count'] for b in bottlenecks)}\n")
                f.write(f"Average Aircraft per Bottleneck: {sum(b['aircraft_count'] for b in bottlenecks) / len(bottlenecks):.1f}\n")
            f.write("\n")
            
            # Write detailed bottleneck analysis
            if bottlenecks:
                f.write("DETAILED TIME-BASED BOTTLENECK ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                for i, bottleneck in enumerate(bottlenecks, 1):
                    f.write(f"\nBOTTLENECK #{i}\n")
                    f.write(f"Type: {bottleneck['bottleneck_type'].replace('_', ' ').title()}\n")
                    f.write(f"Time Window: {bottleneck['time_window'].strftime('%Y-%m-%d %H:%M')}\n")
                    f.write(f"Start Time: {bottleneck['start_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"End Time: {bottleneck['end_time'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Duration: {bottleneck['duration_minutes']:.1f} minutes\n")
                    f.write(f"Location: {bottleneck['coordinates'][0]:.6f}, {bottleneck['coordinates'][1]:.6f}\n")
                    f.write(f"Aircraft Count: {bottleneck['aircraft_count']}\n")
                    
                    f.write(f"\nAircraft Types:\n")
                    for ac_type, count in sorted(bottleneck['aircraft_types'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {ac_type}: {count} aircraft\n")
                    
                    f.write(f"\nAirlines:\n")
                    for airline, count in sorted(bottleneck['airlines'].items(), key=lambda x: x[1], reverse=True):
                        f.write(f"  {airline}: {count} flights\n")
                    
                    f.write(f"\nAffected Flight Numbers:\n")
                    unique_flights = list(set(bottleneck['affected_flights']))
                    for flight in unique_flights:
                        if flight != 'UNKNOWN':
                            f.write(f"  {flight}\n")
                    
                    f.write(f"\nSample Aircraft Positions:\n")
                    for ac in bottleneck['sample_aircraft']:
                        flight_id = ac.get('flight', 'UNKNOWN')
                        ac_type = ac.get('t', 'UNKNOWN')
                        lat = ac.get('lat', 0)
                        lon = ac.get('lon', 0)
                        f.write(f"  {flight_id} ({ac_type}): {lat:.6f}, {lon:.6f}\n")
                    
                    f.write("-" * 40 + "\n")
            else:
                f.write("NO BOTTLENECKS DETECTED\n")
                f.write("-" * 40 + "\n")
                f.write("All time periods had fewer than 2 aircraft\n")
                f.write("No significant congestion detected\n")
            
            # Write footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Time-Based Analysis Report\n")
            f.write("Generated by Airport Bottleneck Prediction System\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✅ Time-based results written to time_based_results.txt")
        
    except Exception as e:
        print(f"❌ Error writing time-based results: {e}")

def main():
    """Main function"""
    print("🕐 TIME-BASED BOTTLENECK ANALYSIS")
    print("=" * 80)
    print("🎯 Detecting bottlenecks based on time periods with 2+ aircraft")
    print("📅 Showing when and where bottlenecks occur")
    print("✈️  Listing specific flight numbers affected")
    
    # Get airport code from command line or use default
    airport_code = "KJFK"
    if len(sys.argv) > 1:
        airport_code = sys.argv[1].upper()
    
    print(f"\n🏢 Analyzing airport: {airport_code}")
    print(f"📁 Looking for data.json in: {os.getcwd()}")
    
    # Generate time-based analysis
    bottlenecks = generate_time_based_analysis(airport_code)
    
    if bottlenecks:
        print(f"\n🎯 SUCCESS!")
        print(f"✅ Time-based analysis complete")
        print(f"📄 Results written to: time_based_results.txt")
        print(f"⏰ Found {len(bottlenecks)} bottleneck time periods")
        print(f"✈️  Each bottleneck shows specific flight numbers affected")
    else:
        print(f"\n✅ NO BOTTLENECKS DETECTED!")
        print(f"💡 All time periods had fewer than 2 aircraft")
        print(f"📄 Results written to: time_based_results.txt")

if __name__ == "__main__":
    main()
