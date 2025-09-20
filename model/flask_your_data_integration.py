"""
Flask Integration for Your New data.json Format

This script integrates your flight tracking data format with your Flask app
to provide real-time bottleneck predictions.
"""

from flask import Flask, render_template, jsonify, request
import json
import os
from datetime import datetime

# Import the updated data loader and model
from data_loader import DataLoader
from simple_airport_model import SimpleAirportBottleneckModel
from simple_config import SIMPLE_BOTTLENECK_CONFIG

app = Flask(__name__)

# Initialize components
data_loader = DataLoader("data.json")
bottleneck_model = SimpleAirportBottleneckModel(SIMPLE_BOTTLENECK_CONFIG)

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/airport/<code>')
def airport(code):
    """Airport page with map"""
    airport_code = code.upper()
    
    # Get airport coordinates
    coords = get_airport_coordinates(airport_code)
    
    # Create sample airport data
    airport_data = {
        'name': f'{airport_code} Airport',
        'city': 'Unknown',
        'country': 'Unknown',
        'coordinates': coords,
        'runways': [
            {'id': '09/27', 'length': 3000, 'width': 45, 'heading': 90},
            {'id': '18/36', 'length': 2500, 'width': 45, 'heading': 180}
        ],
        'taxiways': []
    }
    
    return render_template('airport.html', 
                         airport_code=airport_code, 
                         airport=airport_data)

@app.route('/api/bottlenecks/<code>')
def get_bottlenecks_from_your_data(code):
    """
    API endpoint to get bottleneck predictions from your data.json file
    
    This processes your flight tracking data format!
    """
    airport_code = code.upper()
    
    try:
        print(f"🔄 Processing your data.json for {airport_code}")
        
        # 1. Load your flight tracking data
        raw_data = data_loader.load_data()
        if not raw_data:
            return jsonify({
                'success': False,
                'error': f'Could not load data.json file',
                'airport': airport_code
            })
        
        # 2. Convert your format to ADS-B format
        adsb_data = data_loader.convert_to_adsb_format(raw_data)
        
        # 3. Create airport configuration
        airport_config = {
            'icao': airport_code,
            'coordinates': get_airport_coordinates(airport_code)
        }
        
        # 4. Feed your data into model and get predictions
        print(f"📊 Analyzing {len(adsb_data['aircraft'])} aircraft from your data...")
        analysis = bottleneck_model.predict_bottlenecks(adsb_data, airport_config)
        
        # 5. Write results to results.txt file
        write_results_to_file(analysis, airport_code)
        
        # 6. Return predictions with your data info
        return jsonify({
            'success': True,
            'airport': airport_code,
            'timestamp': datetime.now().isoformat(),
            'data_source': 'Your data.json file',
            'data_date': raw_data.get('date', 'Unknown'),
            'departures_count': len(raw_data.get('departures', [])),
            'arrivals_count': len(raw_data.get('arrivals', [])),
            'total_aircraft': len(adsb_data['aircraft']),
            'bottlenecks': analysis['bottleneck_predictions'],
            'summary': analysis['airport_summary']
        })
        
    except Exception as e:
        print(f"❌ Error processing your data: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'airport': airport_code
        })

@app.route('/api/data-summary/<code>')
def get_data_summary(code):
    """
    API endpoint to get summary of your data.json file
    
    This shows you what's in your flight tracking data
    """
    airport_code = code.upper()
    
    try:
        # Load your data
        raw_data = data_loader.load_data()
        if not raw_data:
            return jsonify({
                'success': False,
                'error': 'Could not load data.json file'
            })
        
        # Analyze your data
        departures = raw_data.get('departures', [])
        arrivals = raw_data.get('arrivals', [])
        
        # Count airlines
        departure_airlines = {}
        for dep in departures:
            callsign = dep.get('callsign', '').strip()
            if callsign:
                airline = callsign[:3]
                departure_airlines[airline] = departure_airlines.get(airline, 0) + 1
        
        arrival_airlines = {}
        for arr in arrivals:
            callsign = arr.get('callsign', '').strip()
            if callsign:
                airline = callsign[:3]
                arrival_airlines[airline] = arrival_airlines.get(airline, 0) + 1
        
        # Count destinations/origins
        destinations = {}
        for dep in departures:
            dest = dep.get('estArrivalAirport')
            if dest:
                destinations[dest] = destinations.get(dest, 0) + 1
        
        origins = {}
        for arr in arrivals:
            origin = arr.get('estDepartureAirport')
            if origin:
                origins[origin] = origins.get(origin, 0) + 1
        
        return jsonify({
            'success': True,
            'airport': airport_code,
            'timestamp': datetime.now().isoformat(),
            'data_date': raw_data.get('date', 'Unknown'),
            'departures': {
                'count': len(departures),
                'top_airlines': dict(list(sorted(departure_airlines.items(), key=lambda x: x[1], reverse=True)[:5])),
                'top_destinations': dict(list(sorted(destinations.items(), key=lambda x: x[1], reverse=True)[:5]))
            },
            'arrivals': {
                'count': len(arrivals),
                'top_airlines': dict(list(sorted(arrival_airlines.items(), key=lambda x: x[1], reverse=True)[:5])),
                'top_origins': dict(list(sorted(origins.items(), key=lambda x: x[1], reverse=True)[:5]))
            },
            'total_flights': len(departures) + len(arrivals)
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/flight-details/<code>')
def get_flight_details(code):
    """
    API endpoint to get detailed flight information from your data
    
    This shows individual flight details
    """
    airport_code = code.upper()
    
    try:
        # Load and convert your data
        raw_data = data_loader.load_data()
        if not raw_data:
            return jsonify({'success': False, 'error': 'Could not load data.json'})
        
        adsb_data = data_loader.convert_to_adsb_format(raw_data)
        aircraft = adsb_data.get('aircraft', [])
        
        # Return sample of flight details
        flight_details = []
        for ac in aircraft[:20]:  # First 20 flights
            flight_details.append({
                'flight': ac.get('flight', 'UNKNOWN'),
                'aircraft_type': ac.get('t', 'UNKNOWN'),
                'icao24': ac.get('icao24', 'UNKNOWN'),
                'flight_type': ac.get('flight_type', 'UNKNOWN'),
                'departure_airport': ac.get('departure_airport'),
                'arrival_airport': ac.get('arrival_airport'),
                'position': [ac.get('lat', 0), ac.get('lon', 0)],
                'altitude': ac.get('alt_baro', 0),
                'heading': ac.get('track', 0),
                'speed': ac.get('gs', 0),
                'timestamp': ac.get('timestamp', 'Unknown')
            })
        
        return jsonify({
            'success': True,
            'airport': airport_code,
            'total_aircraft': len(aircraft),
            'sample_size': len(flight_details),
            'flights': flight_details
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

def write_results_to_file(analysis, airport_code):
    """Write bottleneck analysis results to results.txt file"""
    try:
        with open("results.txt", "w", encoding="utf-8") as f:
            # Write header
            f.write("=" * 80 + "\n")
            f.write("AIRPORT BOTTLENECK ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(f"Airport: {airport_code}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: Simple MLP\n")
            f.write("\n")
            
            # Write summary
            summary = analysis.get('airport_summary', {})
            f.write("AIRPORT SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Aircraft Analyzed: {analysis.get('total_aircraft_monitored', 0)}\n")
            f.write(f"Total Bottlenecks Predicted: {summary.get('total_bottlenecks_predicted', 0)}\n")
            f.write(f"Highest Severity Level: {summary.get('highest_severity_level', 0)}/5\n")
            f.write(f"Total Passengers at Risk: {summary.get('total_passengers_at_risk', 0):,}\n")
            f.write(f"Total Fuel Waste Estimate: {summary.get('total_fuel_waste_estimate', 0):.1f} gallons\n")
            f.write(f"Overall Delay Risk: {summary.get('overall_delay_risk', 'Unknown').upper()}\n")
            f.write("\n")
            
            # Write individual bottlenecks
            bottlenecks = analysis.get('bottleneck_predictions', [])
            if bottlenecks:
                f.write("DETAILED BOTTLENECK ANALYSIS\n")
                f.write("-" * 40 + "\n")
                
                for i, bottleneck in enumerate(bottlenecks, 1):
                    f.write(f"\nBOTTLENECK #{i}\n")
                    f.write(f"Type: {bottleneck.get('type', 'Unknown').replace('_', ' ').upper()}\n")
                    f.write(f"Location: {bottleneck.get('location', {}).get('coordinates', 'Unknown')}\n")
                    f.write(f"Probability: {bottleneck.get('probability', 0):.2f} ({bottleneck.get('probability', 0)*100:.1f}%)\n")
                    f.write(f"Severity: {bottleneck.get('severity', 0)}/5\n")
                    
                    timing = bottleneck.get('timing', {})
                    f.write(f"Predicted Onset: {timing.get('predicted_onset_minutes', 0):.1f} minutes\n")
                    f.write(f"Estimated Duration: {timing.get('estimated_duration_minutes', 0):.1f} minutes\n")
                    f.write(f"Resolution Confidence: {timing.get('resolution_confidence', 0):.2f}\n")
                    
                    impact = bottleneck.get('impact_analysis', {})
                    f.write(f"Passengers Affected: {impact.get('passengers_affected', 0):,}\n")
                    f.write(f"Total Delay: {impact.get('total_delay_minutes', 0):.1f} minutes\n")
                    f.write(f"Fuel Waste: {impact.get('fuel_waste_gallons', 0):.1f} gallons\n")
                    f.write(f"Fuel Cost Estimate: ${impact.get('fuel_cost_estimate', 0):.2f}\n")
                    f.write(f"CO2 Emissions: {impact.get('co2_emissions_lbs', 0):.1f} lbs\n")
                    f.write(f"Economic Impact: ${impact.get('economic_impact_estimate', 0):.2f}\n")
                    
                    mitigations = bottleneck.get('recommended_mitigations', [])
                    if mitigations:
                        f.write(f"Recommended Action: {mitigations[0].get('action', 'None')}\n")
                        f.write(f"Priority: {mitigations[0].get('priority', 'Unknown')}\n")
                        f.write(f"Effectiveness: {mitigations[0].get('estimated_effectiveness', 0):.1f}\n")
                        f.write(f"Implementation Time: {mitigations[0].get('implementation_time', 0):.1f} minutes\n")
                    
                    f.write("-" * 40 + "\n")
            else:
                f.write("NO BOTTLENECKS PREDICTED\n")
                f.write("-" * 40 + "\n")
                f.write("All airport operations are running smoothly!\n")
                f.write("No significant congestion or delays detected.\n")
            
            # Write aircraft details
            f.write("\nAIRCRAFT DETAILS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total Aircraft Monitored: {analysis.get('total_aircraft_monitored', 0)}\n")
            f.write(f"Analysis Radius: {analysis.get('analysis_radius_nm', 3)} nautical miles\n")
            f.write(f"Data Source: {analysis.get('model_info', {}).get('model_type', 'Unknown')}\n")
            
            # Write footer
            f.write("\n" + "=" * 80 + "\n")
            f.write("End of Analysis Report\n")
            f.write("Generated by Airport Bottleneck Prediction System\n")
            f.write("=" * 80 + "\n")
        
        print(f"✅ Results written to results.txt")
        
    except Exception as e:
        print(f"❌ Error writing results to file: {e}")

def get_airport_coordinates(airport_code):
    """Get airport coordinates"""
    airport_coords = {
        'KJFK': (40.63980103, -73.77890015),
        'KLAX': (33.9425, -118.4081),
        'KORD': (41.9786, -87.9048),
        'KDFW': (32.8968, -97.0380),
        'KPHL': (39.8719, -75.2411),
        'KATL': (33.6407, -84.4277),
        'KLGA': (40.7769, -73.8740),
        'KBOS': (42.3656, -71.0096)
    }
    return airport_coords.get(airport_code, (0.0, 0.0))

if __name__ == '__main__':
    print("🚀 Starting Flask app with YOUR data.json integration")
    print("📍 Available endpoints:")
    print("   • /airport/<code> - Airport map page")
    print("   • /api/bottlenecks/<code> - Get bottleneck predictions from your data")
    print("   • /api/data-summary/<code> - Get summary of your flight data")
    print("   • /api/flight-details/<code> - Get detailed flight information")
    print("\n🎯 Example usage:")
    print("   • http://localhost:5000/airport/JFK")
    print("   • http://localhost:5000/api/bottlenecks/JFK")
    print("   • http://localhost:5000/api/data-summary/JFK")
    print("   • http://localhost:5000/api/flight-details/JFK")
    print("\n📊 Your data.json contains:")
    print("   • 344 departures")
    print("   • 505 arrivals") 
    print("   • 849 total flights")
    print("   • Airlines: DAL, JBU, DLH, QTR, THY, BAW, etc.")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
