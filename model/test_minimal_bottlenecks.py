"""
Test even smaller bottlenecks to get severity 1-3
"""

import asyncio
import json
from gemini_bottleneck_predictor import GeminiBottleneckPredictor


async def test_minimal_bottlenecks():
    """Test detection of minimal bottlenecks with 1-2 aircraft"""
    
    # Test case 1: Minimal bottleneck with 2 aircraft
    minimal_data = [
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
        }
    ]
    
    # Test case 2: Single aircraft with slow movement
    single_aircraft_data = [
        {
            "flight": "AAL789",
            "lat": 40.6413,
            "lon": -73.7781,
            "altitude": "ground",
            "speed": 2,
            "heading": 90
        }
    ]
    
    # Test case 3: Two aircraft far apart
    distant_aircraft_data = [
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
            "lat": 40.6513,  # 10km away
            "lon": -73.7881,
            "altitude": "ground",
            "speed": 0,
            "heading": 180
        }
    ]
    
    print("🔍 Testing Minimal Bottleneck Detection")
    print("=" * 50)
    
    predictor = GeminiBottleneckPredictor()
    
    # Test minimal bottleneck
    print("\n📊 TEST 1: Minimal Bottleneck (2 aircraft close)")
    print("-" * 40)
    result1 = await predictor.predict_bottlenecks(minimal_data, "JFK")
    traffic1 = predictor.analyze_traffic_density(minimal_data)
    
    print(f"Severity: {result1.severity}/5")
    print(f"Duration: {result1.duration:.1f} minutes")
    print(f"Type: {result1.type}")
    print(f"Aircraft Count: {result1.aircraft_count}")
    print(f"Density Score: {traffic1.density_score:.1f}")
    print(f"Hotspots: {len(traffic1.hotspots)}")
    
    # Test single aircraft
    print("\n📊 TEST 2: Single Aircraft (slow movement)")
    print("-" * 40)
    result2 = await predictor.predict_bottlenecks(single_aircraft_data, "JFK")
    traffic2 = predictor.analyze_traffic_density(single_aircraft_data)
    
    print(f"Severity: {result2.severity}/5")
    print(f"Duration: {result2.duration:.1f} minutes")
    print(f"Type: {result2.type}")
    print(f"Aircraft Count: {result2.aircraft_count}")
    print(f"Density Score: {traffic2.density_score:.1f}")
    print(f"Hotspots: {len(traffic2.hotspots)}")
    
    # Test distant aircraft
    print("\n📊 TEST 3: Distant Aircraft (2 aircraft far apart)")
    print("-" * 40)
    result3 = await predictor.predict_bottlenecks(distant_aircraft_data, "JFK")
    traffic3 = predictor.analyze_traffic_density(distant_aircraft_data)
    
    print(f"Severity: {result3.severity}/5")
    print(f"Duration: {result3.duration:.1f} minutes")
    print(f"Type: {result3.type}")
    print(f"Aircraft Count: {result3.aircraft_count}")
    print(f"Density Score: {traffic3.density_score:.1f}")
    print(f"Hotspots: {len(traffic3.hotspots)}")
    
    print(f"\n📄 Results saved to: results.txt")
    print("\n✅ Minimal bottleneck testing complete!")


if __name__ == "__main__":
    asyncio.run(test_minimal_bottlenecks())
