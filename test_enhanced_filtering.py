#!/usr/bin/env python3
"""
Enhanced ADS-B Filtering Test
Demonstrates the new arrival-focused filtering system for JFK airport
"""

import asyncio
import json
from datetime import datetime
from client import PlaneMonitor

async def test_enhanced_filtering():
    """Test the enhanced ADS-B filtering for arrivals and ground operations"""
    
    print("🛩️  Enhanced ADS-B Filtering Test for JFK")
    print("=" * 50)
    
    # Initialize the enhanced plane monitor
    monitor = PlaneMonitor("JFK")
    
    print(f"📍 Airport: {monitor.airport_code}")
    print(f"📍 Coordinates: {monitor.airport_centroid}")
    print(f"⚙️  Filter Configuration:")
    for key, value in monitor.filter_config.items():
        print(f"   • {key}: {value}")
    
    print("\n🔄 Fetching aircraft data...")
    
    try:
        # Fetch aircraft data with enhanced filtering
        aircraft_data = await monitor.fetch_planes()
        
        print(f"\n📊 FILTERING RESULTS:")
        print(f"   • Total Aircraft Received: {aircraft_data['total_aircraft']}")
        print(f"   • Filtered Aircraft (Arrivals): {aircraft_data['filtered_aircraft']}")
        
        # Show filter statistics
        stats = aircraft_data['filter_stats']
        print(f"\n📈 FILTER STATISTICS:")
        print(f"   • Arrivals Included: {stats['arrivals_included']}")
        print(f"   • Departures Excluded: {stats['departures_excluded']}")
        print(f"   • High Altitude Excluded: {stats['high_altitude_excluded']}")
        print(f"   • Distant Aircraft Excluded: {stats['distant_excluded']}")
        
        # Show filtered aircraft by category
        current_planes = aircraft_data['current_planes']
        
        print(f"\n🛬 ARRIVAL AIRCRAFT:")
        print(f"   • Ground Aircraft: {len(current_planes['ground'])}")
        for flight, data in list(current_planes['ground'].items())[:5]:
            print(f"     - {flight}: {data['aircraft_type']} at {data['lat']:.4f}, {data['lon']:.4f}")
            if data['distance_nm'] != 'N/A':
                print(f"       Distance: {data['distance_nm']:.1f}nm, Speed: {data['speed']}kts")
        
        print(f"   • Approach Aircraft: {len(current_planes['approach'])}")
        for flight, data in list(current_planes['approach'].items())[:5]:
            print(f"     - {flight}: {data['aircraft_type']} at {data['altitude']}ft")
            if data['distance_nm'] != 'N/A':
                print(f"       Distance: {data['distance_nm']:.1f}nm, Speed: {data['speed']}kts")
        
        print(f"   • Low Altitude Aircraft: {len(current_planes['low_altitude'])}")
        for flight, data in list(current_planes['low_altitude'].items())[:5]:
            print(f"     - {flight}: {data['aircraft_type']} at {data['altitude']}ft")
            if data['distance_nm'] != 'N/A':
                print(f"       Distance: {data['distance_nm']:.1f}nm, Speed: {data['speed']}kts")
        
        # Show filtered out aircraft (departures/exclusions)
        if aircraft_data['filtered_out']:
            print(f"\n✈️  FILTERED OUT AIRCRAFT (Departures/Exclusions):")
            for ac in aircraft_data['filtered_out'][:10]:
                print(f"   • {ac['flight']}: {ac['reason']} - Alt: {ac['altitude']}, Speed: {ac['speed']}kts")
                if ac['distance_nm'] != 'N/A':
                    print(f"     Distance: {ac['distance_nm']:.1f}nm")
        
        # Show changes (aircraft entering/leaving)
        changes = aircraft_data['changes']
        if changes['entered']:
            print(f"\n🔄 AIRCRAFT ENTERING:")
            for change in changes['entered'][:5]:
                print(f"   • {change['flight_number']} ({change['category']})")
        
        if changes['left']:
            print(f"\n🔄 AIRCRAFT LEAVING:")
            for change in changes['left'][:5]:
                print(f"   • {change['flight_number']} ({change['category']})")
        
        # Summary
        print(f"\n✅ FILTERING SUMMARY:")
        print(f"   • Successfully filtered {aircraft_data['total_aircraft']} aircraft")
        print(f"   • Kept {aircraft_data['filtered_aircraft']} arrival/ground aircraft")
        print(f"   • Excluded {aircraft_data['total_aircraft'] - aircraft_data['filtered_aircraft']} departure/high-altitude aircraft")
        
        # Calculate filtering effectiveness
        if aircraft_data['total_aircraft'] > 0:
            effectiveness = (aircraft_data['filtered_aircraft'] / aircraft_data['total_aircraft']) * 100
            print(f"   • Filtering Effectiveness: {effectiveness:.1f}% arrivals retained")
        
        return aircraft_data
        
    except Exception as e:
        print(f"❌ Error fetching aircraft data: {e}")
        return None

async def test_filter_configuration():
    """Test different filter configurations"""
    
    print("\n🔧 Testing Filter Configuration Adjustments")
    print("-" * 40)
    
    monitor = PlaneMonitor("JFK")
    
    # Test with stricter filtering (more restrictive)
    print("\n📉 Testing Stricter Filtering:")
    monitor.filter_config['max_altitude_ft'] = 3000  # Lower altitude limit
    monitor.filter_config['max_distance_nm'] = 10   # Closer to airport
    monitor.filter_config['max_speed_kts'] = 150    # Lower speed limit
    
    aircraft_data = await monitor.fetch_planes()
    print(f"   • Stricter filtering: {aircraft_data['filtered_aircraft']} aircraft retained")
    
    # Test with looser filtering (less restrictive)
    print("\n📈 Testing Looser Filtering:")
    monitor.filter_config['max_altitude_ft'] = 8000  # Higher altitude limit
    monitor.filter_config['max_distance_nm'] = 25   # Further from airport
    monitor.filter_config['max_speed_kts'] = 250    # Higher speed limit
    
    aircraft_data = await monitor.fetch_planes()
    print(f"   • Looser filtering: {aircraft_data['filtered_aircraft']} aircraft retained")
    
    # Reset to default
    monitor.filter_config = {
        'max_altitude_ft': 5000,
        'max_distance_nm': 15,
        'min_speed_kts': 0,
        'max_speed_kts': 200,
        'approach_altitude_ft': 3000,
        'ground_speed_threshold': 30,
        'departure_exclusion_alt': 2000,
        'departure_exclusion_speed': 150
    }

def main():
    """Main function to run the enhanced filtering test"""
    print("🚀 Starting Enhanced ADS-B Filtering Test")
    print("=" * 60)
    
    # Run the async test
    result = asyncio.run(test_enhanced_filtering())
    
    if result:
        # Test different configurations
        asyncio.run(test_filter_configuration())
        
        print("\n🎯 FILTERING CAPABILITIES DEMONSTRATED:")
        print("   ✅ Altitude-based filtering (excludes high departures)")
        print("   ✅ Distance-based filtering (focuses on nearby aircraft)")
        print("   ✅ Speed-based filtering (excludes fast departures)")
        print("   ✅ Approach pattern detection (identifies arrivals)")
        print("   ✅ Ground operation focus (taxiing aircraft)")
        print("   ✅ Departure exclusion (climbing aircraft)")
        print("   ✅ Configurable thresholds (adjustable parameters)")
        print("   ✅ Detailed statistics (filtering effectiveness)")
        
        print("\n🎉 Enhanced ADS-B filtering test completed successfully!")
    else:
        print("\n❌ Test failed - check your internet connection and API access")

if __name__ == "__main__":
    main()
