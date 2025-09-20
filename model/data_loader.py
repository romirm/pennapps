"""
Data Loader for Bottleneck Prediction Model

This module shows you how to load your data.json file and feed it
into the simplified MLP bottleneck prediction model.
"""

import json
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# Import the simplified model
from simple_airport_model import SimpleAirportBottleneckModel
from simple_config import SIMPLE_BOTTLENECK_CONFIG


class DataLoader:
    """
    Loads data from JSON files and feeds it into the bottleneck prediction model
    """
    
    def __init__(self, data_file_path: str = "data.json"):
        self.data_file_path = data_file_path
        self.model = SimpleAirportBottleneckModel(SIMPLE_BOTTLENECK_CONFIG)
        
    def load_data(self) -> Optional[Dict]:
        """
        Load data from your JSON file
        
        Returns:
            Dictionary containing the loaded data, or None if file not found
        """
        try:
            if not os.path.exists(self.data_file_path):
                print(f"❌ Data file not found: {self.data_file_path}")
                return None
                
            with open(self.data_file_path, 'r') as f:
                data = json.load(f)
                
            print(f"✅ Successfully loaded data from {self.data_file_path}")
            return data
            
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def convert_to_adsb_format(self, data: Dict) -> Dict:
        """
        Convert your data format to ADS-B format expected by the model
        
        Args:
            data: Your loaded JSON data
            
        Returns:
            Data in ADS-B format for the model
        """
        # Handle different possible data formats
        if 'aircraft' in data:
            # Already in ADS-B format
            return data
        elif 'departures' in data and 'arrivals' in data:
            # NEW FORMAT: Flight tracking data with departures and arrivals
            aircraft = []
            
            # Process departures
            for departure in data.get('departures', []):
                aircraft.append({
                    'flight': departure.get('callsign', 'UNKNOWN').strip(),
                    't': self._get_aircraft_type_from_icao(departure.get('icao24', '')),
                    'lat': self._estimate_latitude_from_distance(departure.get('estDepartureAirportHorizDistance', 0)),
                    'lon': self._estimate_longitude_from_distance(departure.get('estDepartureAirportHorizDistance', 0)),
                    'alt_baro': departure.get('estDepartureAirportVertDistance', 0) * 100,  # Convert to feet
                    'track': self._estimate_heading_from_airports(departure.get('estDepartureAirport'), departure.get('estArrivalAirport')),
                    'gs': self._estimate_speed_from_flight_data(departure),
                    'timestamp': datetime.fromtimestamp(departure.get('firstSeen', 0)).isoformat(),
                    'icao24': departure.get('icao24', ''),
                    'flight_type': 'departure',
                    'departure_airport': departure.get('estDepartureAirport'),
                    'arrival_airport': departure.get('estArrivalAirport')
                })
            
            # Process arrivals
            for arrival in data.get('arrivals', []):
                aircraft.append({
                    'flight': arrival.get('callsign', 'UNKNOWN').strip(),
                    't': self._get_aircraft_type_from_icao(arrival.get('icao24', '')),
                    'lat': self._estimate_latitude_from_distance(arrival.get('estArrivalAirportHorizDistance', 0)),
                    'lon': self._estimate_longitude_from_distance(arrival.get('estArrivalAirportHorizDistance', 0)),
                    'alt_baro': arrival.get('estArrivalAirportVertDistance', 0) * 100,  # Convert to feet
                    'track': self._estimate_heading_from_airports(arrival.get('estDepartureAirport'), arrival.get('estArrivalAirport')),
                    'gs': self._estimate_speed_from_flight_data(arrival),
                    'timestamp': datetime.fromtimestamp(arrival.get('lastSeen', 0)).isoformat(),
                    'icao24': arrival.get('icao24', ''),
                    'flight_type': 'arrival',
                    'departure_airport': arrival.get('estDepartureAirport'),
                    'arrival_airport': arrival.get('estArrivalAirport')
                })
            
            return {'aircraft': aircraft}
        elif 'flights' in data:
            # Convert flights to aircraft format
            aircraft = []
            for flight in data['flights']:
                aircraft.append({
                    'flight': flight.get('id', 'UNKNOWN'),
                    't': flight.get('aircraft_type', 'UNKNOWN'),
                    'lat': flight.get('latitude', 0),
                    'lon': flight.get('longitude', 0),
                    'alt_baro': flight.get('altitude', 0),
                    'track': flight.get('heading', 0),
                    'gs': flight.get('speed', 0),
                    'timestamp': flight.get('timestamp', datetime.now().isoformat())
                })
            return {'aircraft': aircraft}
        elif 'planes' in data:
            # Convert planes to aircraft format
            aircraft = []
            for plane in data['planes']:
                aircraft.append({
                    'flight': plane.get('flight_number', 'UNKNOWN'),
                    't': plane.get('type', 'UNKNOWN'),
                    'lat': plane.get('lat', 0),
                    'lon': plane.get('lng', 0),
                    'alt_baro': plane.get('alt', 0),
                    'track': plane.get('course', 0),
                    'gs': plane.get('velocity', 0),
                    'timestamp': plane.get('time', datetime.now().isoformat())
                })
            return {'aircraft': aircraft}
        else:
            # Try to convert generic data
            aircraft = []
            if isinstance(data, list):
                for item in data:
                    aircraft.append({
                        'flight': item.get('flight', item.get('id', 'UNKNOWN')),
                        't': item.get('aircraft_type', item.get('type', 'UNKNOWN')),
                        'lat': item.get('lat', item.get('latitude', 0)),
                        'lon': item.get('lon', item.get('longitude', 0)),
                        'alt_baro': item.get('alt', item.get('altitude', 0)),
                        'track': item.get('track', item.get('heading', 0)),
                        'gs': item.get('gs', item.get('speed', 0)),
                        'timestamp': item.get('timestamp', datetime.now().isoformat())
                    })
            return {'aircraft': aircraft}
    
    def _get_aircraft_type_from_icao(self, icao24: str) -> str:
        """Estimate aircraft type from ICAO24 code"""
        if not icao24:
            return 'UNKNOWN'
        
        # Simple mapping based on common ICAO24 patterns
        icao_prefixes = {
            'a': 'B737',  # American Airlines
            'ad': 'A320', # JetBlue
            '3c': 'A320', # Lufthansa
            '06': 'B777', # Qatar Airways
            '4b': 'B737', # Turkish Airlines
            '40': 'A320', # British Airways
            'a0': 'B737', # Republic Airways
        }
        
        for prefix, aircraft_type in icao_prefixes.items():
            if icao24.lower().startswith(prefix):
                return aircraft_type
        
        return 'B737'  # Default to B737
    
    def _estimate_latitude_from_distance(self, distance: int) -> float:
        """Estimate latitude based on distance from airport"""
        if not distance:
            return 40.6398  # JFK default
        
        # Rough estimation: closer flights are near airport center
        # This is a simplified approach - in reality you'd need more sophisticated positioning
        base_lat = 40.6398  # JFK latitude
        lat_offset = (distance / 1000000) * 0.01  # Rough conversion
        return base_lat + lat_offset
    
    def _estimate_longitude_from_distance(self, distance: int) -> float:
        """Estimate longitude based on distance from airport"""
        if not distance:
            return -73.7789  # JFK default
        
        # Rough estimation: closer flights are near airport center
        base_lon = -73.7789  # JFK longitude
        lon_offset = (distance / 1000000) * 0.01  # Rough conversion
        return base_lon + lon_offset
    
    def _estimate_heading_from_airports(self, departure_airport: str, arrival_airport: str) -> float:
        """Estimate heading based on departure and arrival airports"""
        if not departure_airport or not arrival_airport:
            return 90.0  # Default heading
        
        # Simple airport heading mapping
        airport_headings = {
            'KJFK': {'KLAX': 270, 'KORD': 270, 'KDFW': 270, 'KATL': 180, 'KBOS': 45},
            'KLAX': {'KJFK': 90, 'KORD': 90, 'KDFW': 90},
            'KORD': {'KJFK': 90, 'KLAX': 270, 'KDFW': 180},
        }
        
        return airport_headings.get(departure_airport, {}).get(arrival_airport, 90.0)
    
    def _estimate_speed_from_flight_data(self, flight_data: Dict) -> float:
        """Estimate ground speed from flight data"""
        # Estimate speed based on flight type and distance
        if flight_data.get('estArrivalAirport'):
            # International flight
            return 500.0  # knots
        else:
            # Domestic flight
            return 300.0  # knots
    
    def predict_bottlenecks_from_file(self, airport_code: str = "KJFK") -> Optional[Dict]:
        """
        Load data from your JSON file and predict bottlenecks
        
        Args:
            airport_code: ICAO airport code
            
        Returns:
            Bottleneck analysis results
        """
        print(f"🔄 Loading data from {self.data_file_path} for {airport_code}")
        
        # 1. Load your data
        raw_data = self.load_data()
        if not raw_data:
            return None
        
        # 2. Convert to ADS-B format
        adsb_data = self.convert_to_adsb_format(raw_data)
        
        # 3. Create airport configuration
        airport_config = {
            'icao': airport_code,
            'coordinates': self.get_airport_coordinates(airport_code)
        }
        
        # 4. Feed data into model and get predictions
        print(f"📊 Analyzing {len(adsb_data['aircraft'])} aircraft...")
        analysis = self.model.predict_bottlenecks(adsb_data, airport_config)
        
        # 5. Write results to results.txt file
        self.write_results_to_file(analysis, airport_code)
        
        return analysis
    
    def write_results_to_file(self, analysis: Dict, airport_code: str):
        """
        Write bottleneck analysis results to results.txt file
        
        Args:
            analysis: Bottleneck analysis results
            airport_code: ICAO airport code
        """
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
    
    def get_airport_coordinates(self, airport_code: str) -> tuple:
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


def create_sample_data_json():
    """
    Create a sample data.json file to show you the expected format
    """
    import time
    
    current_time = int(time.time())
    
    sample_data = {
        "date": "2025-09-20 16:59:20.603325",
        "departures": [
            {
                "icao24": "a67805",
                "firstSeen": current_time - 3600,
                "estDepartureAirport": "KJFK",
                "lastSeen": current_time - 1800,
                "estArrivalAirport": "KLAX",
                "callsign": "DAL1975",
                "estDepartureAirportHorizDistance": 1286,
                "estDepartureAirportVertDistance": 26,
                "estArrivalAirportHorizDistance": 0,
                "estArrivalAirportVertDistance": 0,
                "departureAirportCandidatesCount": 285,
                "arrivalAirportCandidatesCount": 0
            },
            {
                "icao24": "ad89f7",
                "firstSeen": current_time - 3000,
                "estDepartureAirport": "KJFK",
                "lastSeen": current_time - 1200,
                "estArrivalAirport": "KBOS",
                "callsign": "JBU2637",
                "estDepartureAirportHorizDistance": 1286,
                "estDepartureAirportVertDistance": 34,
                "estArrivalAirportHorizDistance": 0,
                "estArrivalAirportVertDistance": 0,
                "departureAirportCandidatesCount": 285,
                "arrivalAirportCandidatesCount": 0
            }
        ],
        "arrivals": [
            {
                "icao24": "406947",
                "firstSeen": current_time - 7200,
                "estDepartureAirport": "EGLL",
                "lastSeen": current_time - 600,
                "estArrivalAirport": "KJFK",
                "callsign": "BAW33K",
                "estDepartureAirportHorizDistance": 1527,
                "estDepartureAirportVertDistance": 104,
                "estArrivalAirportHorizDistance": 8325,
                "estArrivalAirportVertDistance": 361,
                "departureAirportCandidatesCount": 3,
                "arrivalAirportCandidatesCount": 1
            },
            {
                "icao24": "a05614",
                "firstSeen": current_time - 5400,
                "estDepartureAirport": "KATL",
                "lastSeen": current_time - 300,
                "estArrivalAirport": "KJFK",
                "callsign": "DAL2466",
                "estDepartureAirportHorizDistance": 1149,
                "estDepartureAirportVertDistance": 30,
                "estArrivalAirportHorizDistance": 20027,
                "estArrivalAirportVertDistance": 902,
                "departureAirportCandidatesCount": 1,
                "arrivalAirportCandidatesCount": 1
            }
        ]
    }
    
    with open("sample_data.json", 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print("✅ Created sample data.json file with new format")
    return sample_data


def main():
    """
    Demo function showing how to use your data.json file
    """
    print("🎯 DATA.JSON INTEGRATION DEMO")
    print("=" * 50)
    
    # 1. Create sample data.json if it doesn't exist
    if not os.path.exists("data.json"):
        print("📝 Creating sample data.json file...")
        create_sample_data_json()
    
    # 2. Load data and predict bottlenecks
    loader = DataLoader("data.json")
    analysis = loader.predict_bottlenecks_from_file("KJFK")
    
    if analysis:
        print("\n📋 BOTTLENECK ANALYSIS RESULTS")
        print("=" * 40)
        
        summary = analysis['airport_summary']
        print(f"🏢 Airport: {analysis['airport']}")
        print(f"✈️  Aircraft Analyzed: {analysis['total_aircraft_monitored']}")
        print(f"🚨 Total Bottlenecks: {summary['total_bottlenecks_predicted']}")
        print(f"⚠️  Risk Level: {summary['overall_delay_risk'].upper()}")
        print(f"👥 Passengers at Risk: {summary['total_passengers_at_risk']}")
        print(f"⛽ Fuel Waste: {summary['total_fuel_waste_estimate']:.1f} gallons")
        
        # Show individual bottlenecks
        if analysis['bottleneck_predictions']:
            print(f"\n🚨 INDIVIDUAL BOTTLENECKS:")
            for i, bottleneck in enumerate(analysis['bottleneck_predictions'], 1):
                print(f"\n   {i}. {bottleneck['type'].upper()}")
                print(f"      📍 Location: {bottleneck['location']['coordinates']}")
                print(f"      📊 Probability: {bottleneck['probability']:.2f}")
                print(f"      ⚠️  Severity: {bottleneck['severity']}/5")
                print(f"      ⏱️  Delay: {bottleneck['timing']['estimated_duration_minutes']:.1f} min")
        
        # Save results
        output_file = f"bottleneck_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        print(f"\n💾 Results saved to: {output_file}")
    
    print("\n🎯 HOW TO USE YOUR DATA.JSON:")
    print("1. Put your data.json file in the project root directory")
    print("2. Use DataLoader to load and analyze your data")
    print("3. Get bottleneck predictions with exact coordinates")
    print("4. Display results on your airport map")


if __name__ == "__main__":
    main()
