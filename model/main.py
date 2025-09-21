"""
Main Air Traffic Bottleneck Prediction System
Processes data.json and generates detailed bottleneck analysis reports
"""

import os
import sys
from datetime import datetime
from typing import List, Dict

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flight_processor import AirportDatabase, FlightProcessor, FlightPosition
from bottleneck_analyzer import BottleneckAnalyzer, Bottleneck


class AirTrafficBottleneckSystem:
    """Main system for air traffic bottleneck prediction"""
    
    def __init__(self):
        self.airport_db = AirportDatabase()
        self.flight_processor = FlightProcessor(self.airport_db)
        self.bottleneck_analyzer = BottleneckAnalyzer()
        self.data_file = 'data.json'
    
    def run_analysis(self, data_file: str = 'data.json') -> Dict:
        """Run the complete bottleneck analysis"""
        print("=" * 80)
        print("AIR TRAFFIC BOTTLENECK PREDICTION SYSTEM")
        print("=" * 80)
        print(f"Starting analysis at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Data source: {data_file}")
        print()
        
        # Step 1: Process flight data
        print("Step 1: Processing flight data...")
        flight_positions = self.flight_processor.process_data_json(data_file)
        
        if not flight_positions:
            print("❌ No flight data processed. Check data file.")
            return {}
        
        print(f"✅ Processed {len(flight_positions)} flight positions")
        print()
        
        # Step 2: Analyze bottlenecks
        print("Step 2: Analyzing bottlenecks...")
        analysis_results = self.bottleneck_analyzer.analyze_flights(flight_positions)
        
        print(f"✅ Detected {analysis_results['bottlenecks_detected']} bottlenecks")
        print()
        
        # Step 3: Generate report
        print("Step 3: Generating analysis report...")
        report = self.generate_report(analysis_results)
        
        print("✅ Analysis complete!")
        return analysis_results
    
    def generate_report(self, analysis_results: Dict) -> str:
        """Generate the detailed analysis report in the exact format specified"""
        
        bottlenecks = analysis_results.get('bottlenecks', [])
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("FLIGHT PATH PREDICTION AND BOTTLENECK ANALYSIS RESULTS")
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Data Source: {self.data_file}")
        report_lines.append("")
        
        # Analysis Summary
        report_lines.append("ANALYSIS SUMMARY")
        report_lines.append("-" * 40)
        report_lines.append(f"Total Flight Points: {analysis_results.get('total_flight_points', 0)}")
        report_lines.append(f"Predicted Flight Paths: {analysis_results.get('predicted_paths', 0)}")
        report_lines.append(f"Bottlenecks Detected: {analysis_results.get('bottlenecks_detected', 0)}")
        report_lines.append(f"High Severity Bottlenecks: {analysis_results.get('high_severity_bottlenecks', 0)}")
        report_lines.append("")
        
        # Detailed Bottleneck Analysis
        report_lines.append("DETAILED BOTTLENECK ANALYSIS")
        report_lines.append("-" * 40)
        
        if not bottlenecks:
            report_lines.append("No bottlenecks detected.")
        else:
            for bottleneck in bottlenecks:
                report_lines.append(f"BOTTLENECK #{bottleneck.id}")
                report_lines.append(f"Coordinates: {bottleneck.lat:.6f}, {bottleneck.lng:.6f}")
                report_lines.append(f"Timestamp: {bottleneck.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
                report_lines.append(f"Type: {bottleneck.bottleneck_type}")
                report_lines.append(f"Severity: {bottleneck.severity}/5")
                report_lines.append(f"Duration: {bottleneck.duration_minutes:.1f} minutes")
                report_lines.append(f"Confidence: {bottleneck.confidence:.2f}")
                report_lines.append(f"Aircraft Count: {bottleneck.aircraft_count}")
                report_lines.append("Affected Aircraft:")
                
                # Sort aircraft by callsign for consistent output
                sorted_aircraft = sorted(bottleneck.affected_aircraft, key=lambda x: x.callsign)
                
                for aircraft in sorted_aircraft[:20]:  # Limit to first 20 aircraft
                    report_lines.append(f"  - {aircraft.callsign} ({aircraft.aircraft_type}) - {aircraft.icao24}")
                    report_lines.append(f"    Position: {aircraft.lat:.6f}, {aircraft.lng:.6f}")
                    report_lines.append(f"    Time: {aircraft.timestamp.strftime('%Y-%m-%dT%H:%M:%S')}")
                
                if len(sorted_aircraft) > 20:
                    report_lines.append(f"    ... and {len(sorted_aircraft) - 20} more aircraft")
                
                report_lines.append("")
        
        # Footer
        report_lines.append("=" * 80)
        report_lines.append(f"Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def save_report(self, report: str, filename: str = 'results.txt'):
        """Save the analysis report to file"""
        try:
            with open(filename, 'w') as f:
                f.write(report)
            print(f"📄 Report saved to: {filename}")
        except Exception as e:
            print(f"❌ Error saving report: {e}")
    
    def print_report(self, report: str):
        """Print the analysis report to console"""
        print("\n" + report)
    
    def run_complete_analysis(self, data_file: str = 'data.json', 
                            output_file: str = 'results.txt',
                            print_to_console: bool = True):
        """Run complete analysis and save/print results"""
        
        # Run analysis
        results = self.run_analysis(data_file)
        
        if not results:
            print("❌ Analysis failed - no results generated")
            return
        
        # Generate report
        report = self.generate_report(results)
        
        # Save to file
        self.save_report(report, output_file)
        
        # Print to console if requested
        if print_to_console:
            self.print_report(report)
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Air Traffic Bottleneck Prediction System')
    parser.add_argument('--data', default='data.json', help='Input data file (default: data.json)')
    parser.add_argument('--output', default='results.txt', help='Output report file (default: results.txt)')
    parser.add_argument('--no-console', action='store_true', help='Do not print results to console')
    
    args = parser.parse_args()
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"❌ Error: Data file '{args.data}' not found")
        print("Please ensure the data.json file is in the current directory")
        return
    
    # Run analysis
    system = AirTrafficBottleneckSystem()
    system.run_complete_analysis(
        data_file=args.data,
        output_file=args.output,
        print_to_console=not args.no_console
    )


if __name__ == "__main__":
    main()
