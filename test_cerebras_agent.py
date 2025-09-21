#!/usr/bin/env python3
"""
Test script for the Cerebras Bottleneck Agent
This shows how to use the AgenticBottleneckPredictor to get AI-powered bottleneck analysis
"""

import asyncio
import json
from model.agentic_bottleneck_predictor import AgenticBottleneckPredictor

async def test_cerebras_agent():
    """Test the Cerebras bottleneck prediction agent"""
    
    # Initialize the predictor
    print("🤖 Initializing Cerebras Bottleneck Agent...")
    predictor = AgenticBottleneckPredictor()
    
    # Sample aircraft data (like what you see in your terminal)
    aircraft_data = [
        {
            'flight': 'ACA90',
            'lat': 40.590665,
            'lon': -73.934555,
            'altitude': 0,  # Ground level
            'speed': 0,
            'heading': 0
        },
        {
            'flight': 'UAL123',
            'lat': 40.6413,
            'lon': -73.7781,
            'altitude': 5000,  # In air
            'speed': 180,
            'heading': 90
        },
        {
            'flight': 'DL456',
            'lat': 40.6500,
            'lon': -73.7800,
            'altitude': 0,  # Ground level
            'speed': 15,
            'heading': 45
        }
    ]
    
    airport_code = "JFK"  # John F. Kennedy International Airport
    
    print(f"🛩️  Analyzing {len(aircraft_data)} aircraft at {airport_code}...")
    
    try:
        # This is the main function you call to get Cerebras AI analysis
        analysis_results = await predictor.predict_and_analyze(
            aircraft_data=aircraft_data,
            airport_code=airport_code
        )
        
        print("\n" + "="*60)
        print("🤖 CEREBRAS AI BOTTLENECK ANALYSIS RESULTS")
        print("="*60)
        
        # Display key results
        print(f"📊 Analysis Type: {analysis_results.get('analysis_type', 'Unknown')}")
        print(f"🎯 Confidence Score: {analysis_results.get('confidence_score', 0):.1f}%")
        print(f"🛩️  Airport: {analysis_results.get('airport', 'Unknown')}")
        print(f"⏰ Timestamp: {analysis_results.get('timestamp', 'Unknown')}")
        
        # Traffic Analysis
        if 'traffic_analysis' in analysis_results:
            traffic = analysis_results['traffic_analysis']
            print(f"\n📈 TRAFFIC ANALYSIS:")
            print(f"   • Total Aircraft: {traffic.get('ground_aircraft', 0) + traffic.get('low_alt_aircraft', 0) + traffic.get('high_alt_aircraft', 0)}")
            print(f"   • Ground: {traffic.get('ground_aircraft', 0)}")
            print(f"   • Low Altitude: {traffic.get('low_alt_aircraft', 0)}")
            print(f"   • High Altitude: {traffic.get('high_alt_aircraft', 0)}")
            print(f"   • Density Score: {traffic.get('density_score', 0):.1f}/100")
        
        # Bottleneck Analysis
        if 'bottleneck_analysis' in analysis_results:
            bottleneck = analysis_results['bottleneck_analysis']
            print(f"\n🚧 BOTTLENECK ANALYSIS:")
            print(f"   • AI Analysis: {bottleneck.get('ai_analysis', 'N/A')[:100]}...")
            print(f"   • Bottleneck Count: {bottleneck.get('bottleneck_count', 0)}")
        
        # Impact Analysis
        if 'impact_analysis' in analysis_results:
            impact = analysis_results['impact_analysis']
            print(f"\n💰 IMPACT ANALYSIS:")
            print(f"   • Total Cost Impact: ${impact.get('total_cost_impact', 0):,.2f}")
            print(f"   • Fuel Waste: {impact.get('fuel_waste', 0):.1f} gallons")
            print(f"   • Passengers Affected: {impact.get('passengers_affected', 0)}")
        
        # Save results to file
        predictor.save_results(analysis_results)
        print(f"\n💾 Results saved to model/results.txt")
        
        # Return the full analysis for further use
        return analysis_results
        
    except Exception as e:
        print(f"❌ Error running Cerebras analysis: {e}")
        return None

def main():
    """Main function to run the test"""
    print("🚀 Starting Cerebras Bottleneck Agent Test")
    print("="*50)
    
    # Run the async test
    results = asyncio.run(test_cerebras_agent())
    
    if results:
        print("\n✅ Test completed successfully!")
        print("📋 Full results available in the returned dictionary")
    else:
        print("\n❌ Test failed - check your Cerebras API key configuration")

if __name__ == "__main__":
    main()
