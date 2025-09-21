"""
Demo script for the Air Traffic Bottleneck Prediction System
Shows how to use the system and displays sample results
"""

import os
import sys
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import AirTrafficBottleneckSystem


def demo_system():
    """Demonstrate the bottleneck prediction system"""
    
    print("🚀 Air Traffic Bottleneck Prediction System Demo")
    print("=" * 60)
    print()
    
    # Check if data file exists
    data_file = 'data.json'
    if not os.path.exists(data_file):
        print(f"❌ Error: {data_file} not found")
        print("Please ensure the data.json file is in the current directory")
        return
    
    print(f"📊 Data file found: {data_file}")
    print(f"📁 File size: {os.path.getsize(data_file) / 1024 / 1024:.1f} MB")
    print()
    
    # Initialize system
    print("🔧 Initializing system...")
    system = AirTrafficBottleneckSystem()
    print("✅ System initialized")
    print()
    
    # Run analysis
    print("🔄 Running analysis...")
    start_time = datetime.now()
    
    try:
        results = system.run_analysis(data_file)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"⏱️  Analysis completed in {duration:.1f} seconds")
        print()
        
        # Display summary
        print("📈 ANALYSIS SUMMARY")
        print("-" * 30)
        print(f"Total Flight Points: {results.get('total_flight_points', 0):,}")
        print(f"Predicted Flight Paths: {results.get('predicted_paths', 0):,}")
        print(f"Bottlenecks Detected: {results.get('bottlenecks_detected', 0)}")
        print(f"High Severity Bottlenecks: {results.get('high_severity_bottlenecks', 0)}")
        print()
        
        # Show top bottlenecks
        bottlenecks = results.get('bottlenecks', [])
        if bottlenecks:
            print("🎯 TOP BOTTLENECKS")
            print("-" * 30)
            
            for i, bottleneck in enumerate(bottlenecks[:5]):  # Show top 5
                print(f"{i+1}. {bottleneck.bottleneck_type.upper()} bottleneck")
                print(f"   Location: {bottleneck.lat:.4f}, {bottleneck.lng:.4f}")
                print(f"   Severity: {bottleneck.severity}/5")
                print(f"   Aircraft: {bottleneck.aircraft_count}")
                print(f"   Duration: {bottleneck.duration_minutes:.1f} minutes")
                print(f"   Confidence: {bottleneck.confidence:.2f}")
                print()
        
        # Save results
        output_file = 'demo_results.txt'
        report = system.generate_report(results)
        system.save_report(report, output_file)
        
        print(f"📄 Detailed report saved to: {output_file}")
        print()
        
        # Show sample of detailed report
        print("📋 SAMPLE REPORT OUTPUT")
        print("-" * 30)
        report_lines = report.split('\n')
        for line in report_lines[:20]:  # Show first 20 lines
            print(line)
        print("... (full report in demo_results.txt)")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


def show_usage():
    """Show usage instructions"""
    print("📖 USAGE INSTRUCTIONS")
    print("=" * 30)
    print()
    print("1. Basic usage:")
    print("   python main.py")
    print()
    print("2. Specify custom files:")
    print("   python main.py --data your_data.json --output your_results.txt")
    print()
    print("3. Run without console output:")
    print("   python main.py --no-console")
    print()
    print("4. Programmatic usage:")
    print("   from main import AirTrafficBottleneckSystem")
    print("   system = AirTrafficBottleneckSystem()")
    print("   results = system.run_analysis('data.json')")
    print()


if __name__ == "__main__":
    print("Choose an option:")
    print("1. Run demo analysis")
    print("2. Show usage instructions")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == '1':
        demo_system()
    elif choice == '2':
        show_usage()
    elif choice == '3':
        print("Goodbye!")
    else:
        print("Invalid choice. Running demo...")
        demo_system()
