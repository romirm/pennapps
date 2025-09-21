#!/usr/bin/env python3
"""
Quick interactive test of the Cerebras Bottleneck Agent
"""

import asyncio
from model.agentic_bottleneck_predictor import AgenticBottleneckPredictor

async def quick_test():
    """Quick test with your actual aircraft data"""
    
    # Initialize predictor
    predictor = AgenticBottleneckPredictor()
    
    # Use your actual aircraft data from terminal
    aircraft_data = [
        {
            'flight': 'ACA90',
            'lat': 40.590665,
            'lon': -73.934555,
            'altitude': 0,  # Ground level
            'speed': 0,
            'heading': 0
        }
    ]
    
    print("🛩️  Testing with your aircraft data...")
    print(f"   Aircraft: {aircraft_data[0]['flight']}")
    print(f"   Location: {aircraft_data[0]['lat']}, {aircraft_data[0]['lon']}")
    
    # Run analysis
    results = await predictor.predict_and_analyze(
        aircraft_data=aircraft_data,
        airport_code="JFK"
    )
    
    print("\n📊 RESULTS:")
    print(f"   Analysis Type: {results.get('analysis_type')}")
    print(f"   Confidence: {results.get('confidence_score', 0):.1f}%")
    
    # Show traffic analysis
    if 'traffic_analysis' in results:
        traffic = results['traffic_analysis']
        print(f"   Aircraft Count: {traffic.get('ground_aircraft', 0) + traffic.get('low_alt_aircraft', 0) + traffic.get('high_alt_aircraft', 0)}")
        print(f"   Density Score: {traffic.get('density_score', 0):.1f}/100")
    
    return results

if __name__ == "__main__":
    print("🚀 Quick Cerebras Agent Test")
    print("=" * 30)
    
    results = asyncio.run(quick_test())
    print("\n✅ Test complete!")
