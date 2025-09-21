"""
Run Gemini Bottleneck Predictor and save results
"""

import asyncio
import json
from gemini_bottleneck_predictor import GeminiBottleneckPredictor


async def run_and_save():
    """Run the Gemini predictor and save results"""
    
    # Sample aircraft data
    aircraft_data = [
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
        },
        {
            "flight": "EDV5399",
            "lat": 40.635436,
            "lon": -73.781156,
            "altitude": "ground",
            "speed": 0,
            "heading": 90
        },
        {
            "flight": "RPA5598",
            "lat": 40.633073,
            "lon": -73.776862,
            "altitude": "ground",
            "speed": 0,
            "heading": 180
        },
        {
            "flight": "JBU479",
            "lat": 40.639664,
            "lon": -73.773443,
            "altitude": "ground",
            "speed": 5,
            "heading": 270
        },
        {
            "flight": "JBU83",
            "lat": 40.639966,
            "lon": -73.784330,
            "altitude": "ground",
            "speed": 0,
            "heading": 0
        },
        {
            "flight": "ASA113",
            "lat": 40.642234,
            "lon": -73.764978,
            "altitude": "ground",
            "speed": 0,
            "heading": 90
        }
    ]
    
    print("🚀 Running Gemini Bottleneck Predictor...")
    print(f"📊 Processing {len(aircraft_data)} aircraft")
    
    # Initialize predictor
    predictor = GeminiBottleneckPredictor()
    
    # Run prediction
    result = await predictor.predict_bottlenecks(aircraft_data, "JFK")
    
    # Analyze traffic
    traffic_analysis = predictor.analyze_traffic_density(aircraft_data)
    
    # Save results
    predictor.save_results(result, traffic_analysis, "JFK")
    
    # Display summary
    print("\n📋 PREDICTION RESULTS:")
    print("=" * 50)
    print(f"Bottleneck ID: {result.bottleneck_id}")
    print(f"Coordinates: {result.coordinates}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Type: {result.type}")
    print(f"Severity: {result.severity}/5")
    print(f"Duration: {result.duration:.1f} minutes")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Aircraft Count: {result.aircraft_count}")
    
    print(f"\n✈️  Affected Aircraft:")
    for aircraft in result.aircraft_affected:
        print(f"  - {aircraft['flight_id']} ({aircraft['aircraft_type']})")
        print(f"    Position: {aircraft['position']}")
        print(f"    Time: {aircraft['time']}")
    
    # Convert to JSON format
    result_dict = {
        "bottleneck_id": result.bottleneck_id,
        "coordinates": result.coordinates,
        "timestamp": result.timestamp,
        "type": result.type,
        "severity": result.severity,
        "duration": result.duration,
        "confidence": result.confidence,
        "aircraft_count": result.aircraft_count,
        "aircraft_affected": result.aircraft_affected
    }
    
    # Save JSON to file
    with open('gemini_results.json', 'w') as f:
        json.dump(result_dict, f, indent=2)
    
    print(f"\n📄 JSON results saved to: gemini_results.json")
    print(f"📄 Detailed results saved to: results.txt")
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    asyncio.run(run_and_save())
