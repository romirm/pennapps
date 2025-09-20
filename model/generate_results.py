"""
Simple Script to Generate results.txt

Run this script anytime to analyze your data.json file and generate
a results.txt file with bottleneck predictions.
"""

import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import DataLoader

def generate_results_file(airport_code="KJFK"):
    """
    Generate results.txt file from your data.json
    
    Args:
        airport_code: ICAO airport code to analyze
    """
    print(f"🎯 GENERATING RESULTS.TXT FOR {airport_code}")
    print("=" * 60)
    
    # Check if data.json exists
    if not os.path.exists("model/data.json"):
        print("❌ data.json file not found!")
        print("💡 Make sure your data.json file is in the model directory")
        return False
    
    try:
        # Initialize data loader
        loader = DataLoader("data.json")
        
        # Run analysis and generate results.txt
        print(f"🔄 Analyzing {airport_code} airport...")
        analysis = loader.predict_bottlenecks_from_file(airport_code)
        
        if analysis:
            print(f"✅ Analysis complete!")
            print(f"📄 Results written to: results.txt")
            
            # Show summary
            summary = analysis['airport_summary']
            print(f"\n📊 QUICK SUMMARY:")
            print(f"   • Aircraft Analyzed: {analysis['total_aircraft_monitored']}")
            print(f"   • Bottlenecks Found: {summary['total_bottlenecks_predicted']}")
            print(f"   • Risk Level: {summary['overall_delay_risk'].upper()}")
            print(f"   • Passengers at Risk: {summary['total_passengers_at_risk']:,}")
            print(f"   • Fuel Waste: {summary['total_fuel_waste_estimate']:.1f} gallons")
            
            return True
        else:
            print("❌ Analysis failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def main():
    """Main function"""
    print("🚀 AIRPORT BOTTLENECK ANALYSIS - RESULTS.TXT GENERATOR")
    print("=" * 70)
    
    # Get airport code from command line or use default
    airport_code = "KJFK"
    if len(sys.argv) > 1:
        airport_code = sys.argv[1].upper()
    
    print(f"🏢 Analyzing airport: {airport_code}")
    print(f"📁 Looking for data.json in: {os.getcwd()}")
    
    # Generate results
    success = generate_results_file(airport_code)
    
    if success:
        print(f"\n🎯 SUCCESS!")
        print(f"✅ results.txt file has been created")
        print(f"📖 You can now open results.txt to see detailed bottleneck analysis")
        print(f"🔄 Run this script again anytime to update results.txt with new data")
    else:
        print(f"\n❌ FAILED!")
        print(f"💡 Check that your data.json file is in the correct location")
        print(f"💡 Make sure the file format is correct")

if __name__ == "__main__":
    main()
