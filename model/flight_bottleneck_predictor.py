"""
Flight Data Fetcher and Bottleneck Prediction System

Fetches real flight data from ADS-B.lol API and predicts bottlenecks
using the GNN-KAN model for specific airports and flight numbers.
"""

import requests
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import numpy as np
from model import AirportBottleneckModel
from model.config import BOTTLENECK_CONFIG
from model.adsb_processor import ADSBDataProcessor


class FlightDataFetcher:
    """
    Fetches real flight data from ADS-B.lol API
    """
    
    def __init__(self):
        self.base_url = "https://adsb.lol"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Airport-Bottleneck-Predictor/1.0'
        })
    
    def fetch_aircraft_by_flight_number(self, flight_number: str) -> Optional[Dict]:
        """
        Fetch aircraft data by flight number
        
        Args:
            flight_number: Flight number (e.g., 'UAL123', 'DAL456')
            
        Returns:
            Aircraft data dictionary or None if not found
        """
        try:
            # Clean flight number
            flight_number = flight_number.upper().strip()
            
            # Try different API endpoints
            endpoints = [
                f"/api/flight/{flight_number}",
                f"/api/aircraft/{flight_number}",
                f"/api/track/{flight_number}"
            ]
            
            for endpoint in endpoints:
                try:
                    response = self.session.get(f"{self.base_url}{endpoint}", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        if data and 'aircraft' in data:
                            return data
                        elif isinstance(data, dict) and 'lat' in data:
                            # Single aircraft response
                            return {'aircraft': [data]}
                except Exception as e:
                    print(f"Error fetching from {endpoint}: {e}")
                    continue
            
            print(f"Flight {flight_number} not found in ADS-B.lol")
            return None
            
        except Exception as e:
            print(f"Error fetching flight {flight_number}: {e}")
            return None
    
    def fetch_aircraft_near_airport(self, airport_icao: str, radius_nm: float = 3.0) -> Dict:
        """
        Fetch all aircraft within radius of airport
        
        Args:
            airport_icao: Airport ICAO code (e.g., 'KJFK')
            radius_nm: Search radius in nautical miles
            
        Returns:
            Dictionary containing aircraft data
        """
        try:
            # Get airport coordinates
            airport_coords = self._get_airport_coordinates(airport_icao)
            if not airport_coords:
                return {'aircraft': []}
            
            lat, lon = airport_coords
            
            # Fetch aircraft in area
            url = f"{self.base_url}/api/aircraft"
            params = {
                'lat': lat,
                'lon': lon,
                'radius': radius_nm
            }
            
            response = self.session.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                return data if data else {'aircraft': []}
            else:
                print(f"Error fetching aircraft near {airport_icao}: HTTP {response.status_code}")
                return {'aircraft': []}
                
        except Exception as e:
            print(f"Error fetching aircraft near {airport_icao}: {e}")
            return {'aircraft': []}
    
    def fetch_multiple_flights(self, flight_numbers: List[str]) -> Dict:
        """
        Fetch data for multiple flights
        
        Args:
            flight_numbers: List of flight numbers
            
        Returns:
            Dictionary containing all aircraft data
        """
        all_aircraft = []
        
        for flight_number in flight_numbers:
            print(f"Fetching data for {flight_number}...")
            flight_data = self.fetch_aircraft_by_flight_number(flight_number)
            
            if flight_data and 'aircraft' in flight_data:
                for aircraft in flight_data['aircraft']:
                    # Add flight number if not present
                    if 'flight' not in aircraft or not aircraft['flight']:
                        aircraft['flight'] = flight_number
                    all_aircraft.append(aircraft)
            
            # Rate limiting
            time.sleep(0.5)
        
        return {'aircraft': all_aircraft}
    
    def _get_airport_coordinates(self, airport_icao: str) -> Optional[Tuple[float, float]]:
        """Get airport coordinates"""
        airport_coords = {
            'KJFK': (40.6413, -73.7781),
            'KLAX': (33.9425, -118.4081),
            'KPHL': (39.8729, -75.2407),
            'KMIA': (25.7959, -80.2870),
            'KORD': (41.9786, -87.9048),
            'KDFW': (32.8968, -97.0380),
            'KATL': (33.6407, -84.4277),
            'KDEN': (39.8561, -104.6737),
            'KSFO': (37.6213, -122.3790),
            'KBOS': (42.3656, -71.0096),
            'KSEA': (47.4502, -122.3088),
            'KLAS': (36.0840, -115.1537),
            'KMCO': (28.4312, -81.3081),
            'LHR': (51.4700, -0.4543),
            'CDG': (49.0097, 2.5479),
            'NRT': (35.7720, 140.3928),
            'ICN': (37.4602, 126.4407),
            'DXB': (25.2532, 55.3657),
            'SIN': (1.3644, 103.9915),
            'HKG': (22.3080, 113.9185)
        }
        return airport_coords.get(airport_icao)


class BottleneckPredictor:
    """
    Main class for predicting bottlenecks using real flight data
    """
    
    def __init__(self):
        self.flight_fetcher = FlightDataFetcher()
        self.model = AirportBottleneckModel(BOTTLENECK_CONFIG)
        self.adsb_processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
    
    def predict_bottlenecks_for_flights(self, flight_numbers: List[str], 
                                      airport_icao: str) -> Dict:
        """
        Predict bottlenecks for specific flights at an airport
        
        Args:
            flight_numbers: List of flight numbers to track
            airport_icao: Airport ICAO code
            
        Returns:
            Bottleneck prediction results
        """
        print(f"🛫 Predicting bottlenecks for flights at {airport_icao}")
        print(f"Tracking flights: {', '.join(flight_numbers)}")
        print("=" * 60)
        
        # Fetch flight data
        print("📡 Fetching flight data...")
        adsb_data = self.flight_fetcher.fetch_multiple_flights(flight_numbers)
        
        if not adsb_data.get('aircraft'):
            print("❌ No aircraft data found")
            return self._create_no_data_response(airport_icao, flight_numbers)
        
        print(f"✅ Found {len(adsb_data['aircraft'])} aircraft")
        
        # Display aircraft information
        self._display_aircraft_info(adsb_data['aircraft'])
        
        # Create airport configuration
        airport_config = self._create_airport_config(airport_icao)
        
        # Predict bottlenecks
        print("\n🔍 Analyzing bottlenecks...")
        try:
            analysis = self.model.predict_bottlenecks(adsb_data, airport_config)
            print("✅ Bottleneck analysis completed")
            
            # Display results
            self._display_analysis_results(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return self._create_error_response(airport_icao, str(e))
    
    def predict_bottlenecks_near_airport(self, airport_icao: str, 
                                       radius_nm: float = 3.0) -> Dict:
        """
        Predict bottlenecks for all aircraft near an airport
        
        Args:
            airport_icao: Airport ICAO code
            radius_nm: Search radius in nautical miles
            
        Returns:
            Bottleneck prediction results
        """
        print(f"🛫 Predicting bottlenecks near {airport_icao}")
        print(f"Search radius: {radius_nm} nautical miles")
        print("=" * 60)
        
        # Fetch aircraft data
        print("📡 Fetching aircraft data...")
        adsb_data = self.flight_fetcher.fetch_aircraft_near_airport(airport_icao, radius_nm)
        
        if not adsb_data.get('aircraft'):
            print("❌ No aircraft found near airport")
            return self._create_no_data_response(airport_icao, [])
        
        print(f"✅ Found {len(adsb_data['aircraft'])} aircraft")
        
        # Display aircraft information
        self._display_aircraft_info(adsb_data['aircraft'])
        
        # Create airport configuration
        airport_config = self._create_airport_config(airport_icao)
        
        # Predict bottlenecks
        print("\n🔍 Analyzing bottlenecks...")
        try:
            analysis = self.model.predict_bottlenecks(adsb_data, airport_config)
            print("✅ Bottleneck analysis completed")
            
            # Display results
            self._display_analysis_results(analysis)
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            return self._create_error_response(airport_icao, str(e))
    
    def _display_aircraft_info(self, aircraft_list: List[Dict]):
        """Display information about tracked aircraft"""
        print(f"\n📋 Aircraft Information:")
        print("-" * 40)
        
        for i, aircraft in enumerate(aircraft_list, 1):
            flight_id = aircraft.get('flight', 'UNKNOWN')
            aircraft_type = aircraft.get('t', 'UNKNOWN')
            lat = aircraft.get('lat', 0)
            lon = aircraft.get('lon', 0)
            alt = aircraft.get('alt_baro', 0)
            speed = aircraft.get('gs', 0)
            heading = aircraft.get('track', 0)
            
            print(f"{i}. {flight_id} ({aircraft_type})")
            print(f"   Position: {lat:.4f}, {lon:.4f}")
            print(f"   Altitude: {alt} ft, Speed: {speed} kts, Heading: {heading}°")
    
    def _display_analysis_results(self, analysis: Dict):
        """Display bottleneck analysis results"""
        print(f"\n📊 BOTTLENECK ANALYSIS RESULTS")
        print("=" * 50)
        
        # Airport summary
        summary = analysis.get('airport_summary', {})
        print(f"Airport: {analysis.get('airport', 'UNKNOWN')}")
        print(f"Aircraft monitored: {analysis.get('total_aircraft_monitored', 0)}")
        print(f"Bottlenecks predicted: {summary.get('total_bottlenecks_predicted', 0)}")
        print(f"Highest severity: {summary.get('highest_severity_level', 1)}/5")
        print(f"Overall delay risk: {summary.get('overall_delay_risk', 'low').upper()}")
        print(f"Passengers at risk: {summary.get('total_passengers_at_risk', 0)}")
        print(f"Fuel waste estimate: {summary.get('total_fuel_waste_estimate', 0):.1f} gallons")
        
        # Bottleneck details
        bottlenecks = analysis.get('bottleneck_predictions', [])
        if bottlenecks:
            print(f"\n🚨 BOTTLENECK DETAILS")
            print("-" * 30)
            
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"\nBottleneck #{i}: {bottleneck['type'].replace('_', ' ').title()}")
                print(f"  Location: {bottleneck['location']['zone']}")
                print(f"  Probability: {bottleneck['probability']:.2f}")
                print(f"  Severity: {bottleneck['severity']}/5")
                print(f"  Duration: {bottleneck['timing']['estimated_duration_minutes']:.1f} minutes")
                print(f"  Aircraft affected: {len(bottleneck['aircraft_affected'])}")
                
                impact = bottleneck['impact_analysis']
                print(f"  Passengers affected: {impact['passengers_affected']}")
                print(f"  Fuel waste: {impact['fuel_waste_gallons']:.1f} gallons")
                print(f"  Fuel cost: ${impact['fuel_cost_estimate']:.2f}")
                print(f"  CO2 emissions: {impact['co2_emissions_lbs']:.1f} lbs")
                print(f"  Economic impact: ${impact['economic_impact_estimate']:.2f}")
                
                if bottleneck.get('recommended_mitigations'):
                    print(f"  Recommended actions:")
                    for mitigation in bottleneck['recommended_mitigations'][:2]:
                        print(f"    • {mitigation['action']} (effectiveness: {mitigation['estimated_effectiveness']:.1f})")
        else:
            print("\n✅ No bottlenecks detected - operations running smoothly!")
    
    def _create_airport_config(self, airport_icao: str) -> Dict:
        """Create airport configuration"""
        airport_configs = {
            'KJFK': {
                'icao': 'KJFK',
                'name': 'John F. Kennedy International Airport',
                'city': 'New York',
                'country': 'USA',
                'runways': [
                    {'id': '09L/27R', 'length': 4423, 'width': 45, 'heading': 90},
                    {'id': '09R/27L', 'length': 4423, 'width': 45, 'heading': 90},
                    {'id': '04L/22R', 'length': 3682, 'width': 45, 'heading': 40},
                    {'id': '04R/22L', 'length': 2560, 'width': 45, 'heading': 40}
                ],
                'gates': ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2'],
                'taxiways': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            },
            'KLAX': {
                'icao': 'KLAX',
                'name': 'Los Angeles International Airport',
                'city': 'Los Angeles',
                'country': 'USA',
                'runways': [
                    {'id': '06L/24R', 'length': 2716, 'width': 45, 'heading': 60},
                    {'id': '06R/24L', 'length': 2716, 'width': 45, 'heading': 60},
                    {'id': '07L/25R', 'length': 3658, 'width': 45, 'heading': 70},
                    {'id': '07R/25L', 'length': 3658, 'width': 45, 'heading': 70}
                ],
                'gates': ['A1', 'A2', 'B1', 'B2', 'C1', 'C2', 'D1', 'D2'],
                'taxiways': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
            }
        }
        
        return airport_configs.get(airport_icao, {
            'icao': airport_icao,
            'name': f'{airport_icao} Airport',
            'city': 'Unknown',
            'country': 'Unknown',
            'runways': [
                {'id': '09/27', 'length': 3000, 'width': 45, 'heading': 90},
                {'id': '18/36', 'length': 2500, 'width': 45, 'heading': 180}
            ],
            'gates': ['A1', 'A2', 'B1', 'B2'],
            'taxiways': ['A', 'B', 'C', 'D']
        })
    
    def _create_no_data_response(self, airport_icao: str, flight_numbers: List[str]) -> Dict:
        """Create response when no data is found"""
        return {
            'timestamp': datetime.now().isoformat(),
            'airport': airport_icao,
            'flight_numbers': flight_numbers,
            'status': 'no_data',
            'message': 'No aircraft data found',
            'bottleneck_predictions': [],
            'airport_summary': {
                'total_bottlenecks_predicted': 0,
                'highest_severity_level': 1,
                'total_passengers_at_risk': 0,
                'total_fuel_waste_estimate': 0,
                'overall_delay_risk': 'low'
            }
        }
    
    def _create_error_response(self, airport_icao: str, error_message: str) -> Dict:
        """Create response when error occurs"""
        return {
            'timestamp': datetime.now().isoformat(),
            'airport': airport_icao,
            'status': 'error',
            'error_message': error_message,
            'bottleneck_predictions': [],
            'airport_summary': {
                'total_bottlenecks_predicted': 0,
                'highest_severity_level': 1,
                'total_passengers_at_risk': 0,
                'total_fuel_waste_estimate': 0,
                'overall_delay_risk': 'low'
            }
        }


def main():
    """Main function to demonstrate bottleneck prediction"""
    print("🚀 GNN-KAN Airport Bottleneck Prediction System")
    print("Real-time Flight Data Integration")
    print("=" * 60)
    
    # Initialize predictor
    predictor = BottleneckPredictor()
    
    # Example 1: Predict bottlenecks for specific flights
    print("\n📋 Example 1: Predicting bottlenecks for specific flights")
    print("-" * 60)
    
    # Example flight numbers (these may not be active - replace with real ones)
    example_flights = ['UAL123', 'DAL456', 'SWA789']
    airport = 'KJFK'
    
    print(f"Tracking flights: {', '.join(example_flights)} at {airport}")
    
    # Note: This will likely return "no data" since these are example flights
    # In real usage, you would provide actual active flight numbers
    result1 = predictor.predict_bottlenecks_for_flights(example_flights, airport)
    
    # Example 2: Predict bottlenecks for all aircraft near airport
    print(f"\n📋 Example 2: Predicting bottlenecks for all aircraft near {airport}")
    print("-" * 60)
    
    result2 = predictor.predict_bottlenecks_near_airport(airport, radius_nm=3.0)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    with open(f'bottleneck_prediction_flights_{timestamp}.json', 'w') as f:
        json.dump(result1, f, indent=2, default=str)
    
    with open(f'bottleneck_prediction_airport_{timestamp}.json', 'w') as f:
        json.dump(result2, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to files with timestamp: {timestamp}")
    
    print(f"\n🎯 Usage Instructions:")
    print("1. Replace example flight numbers with real active flights")
    print("2. Use actual airport ICAO codes (KJFK, KLAX, KPHL, etc.)")
    print("3. The system will fetch real-time data from ADS-B.lol")
    print("4. Bottleneck predictions will be generated based on current aircraft positions")


if __name__ == "__main__":
    main()
