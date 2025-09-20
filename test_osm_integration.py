#!/usr/bin/env python3
"""
Test script to demonstrate OpenStreetMap airport data integration
"""

from airport_data_fetcher import get_airport_data_for_app, plot_airport_map
import json

def test_airport_data(airport_code):
    """Test OSM data fetching for a specific airport"""
    print(f"\n{'='*60}")
    print(f"🛫 TESTING AIRPORT DATA FOR: {airport_code}")
    print(f"{'='*60}")
    
    # Get airport data
    data = get_airport_data_for_app(airport_code)
    
    if data:
        print(f"✅ SUCCESS: Found data for {airport_code}")
        print(f"📍 Airport: {data['name']}")
        print(f"🏃 Runways: {len(data['runways'])}")
        print(f"🛣️  Taxiways: {len(data['taxiways'])}")
        
        # Show runway details
        print(f"\n📋 RUNWAY DETAILS:")
        for i, runway in enumerate(data['runways'][:5], 1):  # Show first 5
            print(f"  {i}. {runway['id']}: {runway['length']}m x {runway['width']}m, heading {runway['heading']}°")
        
        if len(data['runways']) > 5:
            print(f"  ... and {len(data['runways']) - 5} more runways")
        
        # Show taxiway details
        print(f"\n🛣️  TAXIWAY DETAILS (first 10):")
        for i, taxiway in enumerate(data['taxiways'][:10], 1):
            print(f"  {i}. {taxiway['id']}: {len(taxiway['points'])} points")
        
        if len(data['taxiways']) > 10:
            print(f"  ... and {len(data['taxiways']) - 10} more taxiways")
        
        # Show metadata
        if 'metadata' in data:
            print(f"\n📊 METADATA:")
            print(f"  Source: {data['metadata'].get('source', 'Unknown')}")
            print(f"  Search Radius: {data['metadata'].get('radius_meters', 'Unknown')}m")
        
        return True
    else:
        print(f"❌ FAILED: No data found for {airport_code}")
        return False

def main():
    """Main test function"""
    print("🚀 OPENSTREETMAP AIRPORT DATA INTEGRATION TEST")
    print("=" * 60)
    
    # Test airports
    test_airports = ['JFK', 'LAX', 'PHL', 'MIA', 'ORD', 'DFW', 'ATL']
    
    results = {}
    
    for airport in test_airports:
        success = test_airport_data(airport)
        results[airport] = success
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 SUMMARY")
    print(f"{'='*60}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    
    print(f"\n🏆 SUCCESS RATE: {(successful/total)*100:.1f}%")
    
    if successful > 0:
        print(f"\n🎉 OSM Integration is working! Your app will now fetch real airport data.")
        print(f"🌐 Data source: OpenStreetMap (free, no API key required)")
        print(f"📈 Benefits:")
        print(f"   • Real runway and taxiway data")
        print(f"   • Up-to-date airport layouts")
        print(f"   • Works for any publicly mapped airport")
        print(f"   • No API limits or costs")
    else:
        print(f"\n⚠️  OSM Integration needs debugging")

if __name__ == "__main__":
    main()
