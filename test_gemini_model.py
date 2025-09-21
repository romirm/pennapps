"""
Test script for the new Gemini-based bottleneck predictor
"""

import asyncio
import json
from model.gemini_bottleneck_predictor import GeminiBottleneckPredictor


async def test_gemini_predictor():
    """Test the Gemini bottleneck predictor with sample data"""
    
    # Sample aircraft data for testing
    test_aircraft = [
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
            "lat": 40.6420,
            "lon": -73.7790,
            "altitude": "ground",
            "speed": 0,
            "heading": 180
        },
        {
            "flight": "AAL789",
            "lat": 40.6415,
            "lon": -73.7785,
            "altitude": "ground",
            "speed": 5,
            "heading": 270
        },
        {
            "flight": "JBU101",
            "lat": 40.6425,
            "lon": -73.7795,
            "altitude": 1500,
            "speed": 120,
            "heading": 45
        },
        {
            "flight": "SWA202",
            "lat": 40.6408,
            "lon": -73.7775,
            "altitude": "ground",
            "speed": 0,
            "heading": 0
        }
    ]
    
    print("🧪 Testing Gemini Bottleneck Predictor...")
    print(f"📊 Test data: {len(test_aircraft)} aircraft")
    
    try:
        # Check if API key is available
        import os
        if not os.getenv('GEMINI_API_KEY'):
            print("⚠️  GEMINI_API_KEY not set. Using fallback mode.")
            print("   Set your API key: export GEMINI_API_KEY='your-api-key'")
        
        # Initialize predictor
        predictor = GeminiBottleneckPredictor()
        
        # Run prediction
        print("🤖 Running prediction...")
        result = await predictor.predict_bottlenecks(test_aircraft, "JFK")
        
        # Display results
        print("\n📋 PREDICTION RESULTS:")
        print(f"   Severity Score: {result.severity_score:.1f}/100")
        print(f"   Risk Level: {result.risk_level}")
        print(f"   Estimated Duration: {result.estimated_duration}")
        print(f"   Confidence: {result.confidence:.1f}%")
        
        if result.bottleneck_locations:
            print(f"\n📍 Bottleneck Locations:")
            for location in result.bottleneck_locations:
                print(f"   - {location}")
        
        if result.affected_aircraft:
            print(f"\n✈️  Affected Aircraft:")
            for aircraft in result.affected_aircraft:
                print(f"   - {aircraft}")
        
        if result.recommendations:
            print(f"\n💡 Recommendations:")
            for rec in result.recommendations:
                print(f"   - {rec}")
        
        print(f"\n✅ Test completed successfully!")
        print(f"📄 Results saved to: {predictor.results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


async def test_traffic_analysis():
    """Test the traffic analysis functionality"""
    
    print("\n🔍 Testing Traffic Analysis...")
    
    # Test with empty data
    predictor = GeminiBottleneckPredictor()
    empty_analysis = predictor.analyze_traffic_density([])
    
    print(f"   Empty data test: {empty_analysis.total_aircraft} aircraft")
    
    # Test with sample data
    sample_data = [
        {"flight": "TEST1", "lat": 40.6413, "lon": -73.7781, "altitude": "ground", "speed": 0, "heading": 90},
        {"flight": "TEST2", "lat": 40.6414, "lon": -73.7782, "altitude": "ground", "speed": 0, "heading": 90},
        {"flight": "TEST3", "lat": 40.6415, "lon": -73.7783, "altitude": "ground", "speed": 0, "heading": 90},
    ]
    
    analysis = predictor.analyze_traffic_density(sample_data)
    
    print(f"   Sample data test: {analysis.total_aircraft} aircraft")
    print(f"   Ground aircraft: {analysis.ground_aircraft}")
    print(f"   Density score: {analysis.density_score:.1f}")
    print(f"   Hotspots: {len(analysis.hotspots)}")
    
    return True


async def main():
    """Run all tests"""
    print("🚀 Starting Gemini Bottleneck Predictor Tests\n")
    
    # Test traffic analysis
    await test_traffic_analysis()
    
    # Test full prediction
    success = await test_gemini_predictor()
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n💥 Some tests failed!")


if __name__ == "__main__":
    asyncio.run(main())
