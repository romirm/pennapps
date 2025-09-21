"""
Test for severity 1 bottlenecks
"""

import asyncio
import json
from gemini_bottleneck_predictor import GeminiBottleneckPredictor


async def test_severity_1():
    """Test detection of severity 1 bottlenecks"""
    
    # Test case: Very minimal bottleneck with 1 aircraft moving slowly
    minimal_data = [
        {
            "flight": "TEST001",
            "lat": 40.6413,
            "lon": -73.7781,
            "altitude": "ground",
            "speed": 1,  # Very slow
            "heading": 90
        }
    ]
    
    print("🔍 Testing Severity 1 Bottleneck Detection")
    print("=" * 50)
    
    predictor = GeminiBottleneckPredictor()
    
    # Test minimal bottleneck
    print("\n📊 TEST: Severity 1 Bottleneck (1 aircraft, very slow)")
    print("-" * 50)
    result = await predictor.predict_bottlenecks(minimal_data, "JFK")
    traffic = predictor.analyze_traffic_density(minimal_data)
    
    print(f"Severity: {result.severity}/5")
    print(f"Duration: {result.duration:.1f} minutes")
    print(f"Type: {result.type}")
    print(f"Aircraft Count: {result.aircraft_count}")
    print(f"Density Score: {traffic.density_score:.1f}")
    print(f"Hotspots: {len(traffic.hotspots)}")
    
    print(f"\n📄 Results saved to: results.txt")
    print("\n✅ Severity 1 testing complete!")


if __name__ == "__main__":
    asyncio.run(test_severity_1())
