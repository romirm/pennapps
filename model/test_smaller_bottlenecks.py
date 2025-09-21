"""
Test smaller bottlenecks with fewer aircraft
"""

import asyncio
import json
from gemini_bottleneck_predictor import GeminiBottleneckPredictor


async def test_smaller_bottlenecks():
    """Test detection of smaller bottlenecks with fewer aircraft"""
    
    # Test case 1: Small bottleneck with 3 aircraft
    small_bottleneck_data = [
        {
            "flight": "DAL123",
            "lat": 40.6413,
            "lon": -73.7781,
            "altitude": "ground",
            "speed": 0,
            "heading": 90
        },
        {
            "flight": "UAL456",
            "lat": 40.6415,
            "lon": -73.7783,
            "altitude": "ground",
            "speed": 5,
            "heading": 180
        },
        {
            "flight": "AAL789",
            "lat": 40.6417,
            "lon": -73.7785,
            "altitude": "ground",
            "speed": 0,
            "heading": 270
        }
    ]
    
    # Test case 2: Medium bottleneck with 5 aircraft
    medium_bottleneck_data = [
        {
            "flight": "DAL123",
            "lat": 40.6413,
            "lon": -73.7781,
            "altitude": "ground",
            "speed": 0,
            "heading": 90
        },
        {
            "flight": "UAL456",
            "lat": 40.6415,
            "lon": -73.7783,
            "altitude": "ground",
            "speed": 5,
            "heading": 180
        },
        {
            "flight": "AAL789",
            "lat": 40.6417,
            "lon": -73.7785,
            "altitude": "ground",
            "speed": 0,
            "heading": 270
        },
        {
            "flight": "JBU101",
            "lat": 40.6419,
            "lon": -73.7787,
            "altitude": "ground",
            "speed": 10,
            "heading": 0
        },
        {
            "flight": "SWA202",
            "lat": 40.6421,
            "lon": -73.7789,
            "altitude": "ground",
            "speed": 0,
            "heading": 45
        }
    ]
    
    print("🔍 Testing Smaller Bottleneck Detection")
    print("=" * 50)
    
    predictor = GeminiBottleneckPredictor()
    
    # Test small bottleneck
    print("\n📊 TEST 1: Small Bottleneck (3 aircraft)")
    print("-" * 30)
    result1 = await predictor.predict_bottlenecks(small_bottleneck_data, "JFK")
    traffic1 = predictor.analyze_traffic_density(small_bottleneck_data)
    
    print(f"Severity: {result1.severity}/5")
    print(f"Duration: {result1.duration:.1f} minutes")
    print(f"Type: {result1.type}")
    print(f"Aircraft Count: {result1.aircraft_count}")
    print(f"Density Score: {traffic1.density_score:.1f}")
    print(f"Hotspots: {len(traffic1.hotspots)}")
    
    # Test medium bottleneck
    print("\n📊 TEST 2: Medium Bottleneck (5 aircraft)")
    print("-" * 30)
    result2 = await predictor.predict_bottlenecks(medium_bottleneck_data, "JFK")
    traffic2 = predictor.analyze_traffic_density(medium_bottleneck_data)
    
    print(f"Severity: {result2.severity}/5")
    print(f"Duration: {result2.duration:.1f} minutes")
    print(f"Type: {result2.type}")
    print(f"Aircraft Count: {result2.aircraft_count}")
    print(f"Density Score: {traffic2.density_score:.1f}")
    print(f"Hotspots: {len(traffic2.hotspots)}")
    
    # Save results
    predictor.save_results(result2, traffic2, "JFK")
    
    print(f"\n📄 Results saved to: results.txt")
    print("\n✅ Smaller bottleneck testing complete!")


if __name__ == "__main__":
    asyncio.run(test_smaller_bottlenecks())
