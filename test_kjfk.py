#!/usr/bin/env python3

import asyncio
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from client import PlaneMonitor
    print("✅ Successfully imported PlaneMonitor")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

async def fetch_kjfk_planes():
    """Fetch current aircraft at or near JFK"""
    print("🛩️ Initializing JFK aircraft monitor...")
    
    monitor = PlaneMonitor()
    result = await monitor.fetch_planes()
    
    print(f"\n📊 JFK Aircraft Status - {result['timestamp']}")
    print(f"Total aircraft: {result['total_aircraft']}")
    
    if result.get('error'):
        print(f"❌ Error: {result['error']}")
        return result
    
    # Display aircraft in air
    if result['current_planes']['air']:
        print(f"\n✈️ Aircraft in air: {len(result['current_planes']['air'])}")
        for flight, data in result['current_planes']['air'].items():
            print(f"   {flight} ({data['aircraft_type']}) - {data['speed']} kts, {data['altitude']} ft")
    else:
        print("\n✈️ No aircraft in air detected")
    
    # Display aircraft on ground
    if result['current_planes']['ground']:
        print(f"\n🛬 Aircraft on ground: {len(result['current_planes']['ground'])}")
        for flight, data in result['current_planes']['ground'].items():
            print(f"   {flight} ({data['aircraft_type']}) - {data['speed']} kts")
    else:
        print("\n🛬 No aircraft on ground detected")
    
    # Display any changes
    if result['changes']['entered']:
        print(f"\n🟢 Aircraft that entered JFK area:")
        for change in result['changes']['entered']:
            print(f"   + {change['flight_number']} ({change['category']})")
    
    if result['changes']['left']:
        print(f"\n🔴 Aircraft that left JFK area:")
        for change in result['changes']['left']:
            print(f"   - {change['flight_number']} ({change['category']})")
    
    return result

if __name__ == "__main__":
    print("🚁 Testing JFK Aircraft Fetching System")
    print("=" * 50)
    
    try:
        result = asyncio.run(fetch_kjfk_planes())
        print(f"\n📈 Summary: {len(result['current_planes']['air'])} air + {len(result['current_planes']['ground'])} ground aircraft")
        
        if result['total_aircraft'] > 0:
            print("✅ Successfully fetched aircraft data!")
        else:
            print("ℹ️ No aircraft detected (this may be normal depending on time/conditions)")
            
    except Exception as e:
        print(f"❌ Error running aircraft fetch: {e}")
        import traceback
        traceback.print_exc()
