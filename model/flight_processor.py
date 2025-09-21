"""
Air Traffic Bottleneck Prediction System
Core classes for processing flight data and detecting bottlenecks
"""

import json
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.cluster import DBSCAN


@dataclass
class FlightPosition:
    """Represents a flight position at a specific time"""
    callsign: str
    icao24: str
    lat: float
    lng: float
    altitude: float
    timestamp: datetime
    aircraft_type: str
    airport: str


@dataclass
class Bottleneck:
    """Represents a detected bottleneck"""
    id: int
    lat: float
    lng: float
    timestamp: datetime
    bottleneck_type: str
    severity: int
    duration_minutes: float
    confidence: float
    aircraft_count: int
    affected_aircraft: List[FlightPosition]


class AirportDatabase:
    """Database of major airport coordinates"""
    
    def __init__(self):
        self.airports = {
            'KJFK': (40.6413, -73.7781),
            'KLGA': (40.7769, -73.8740),
            'KEWR': (40.6895, -74.1745),
            'KBOS': (42.3656, -71.0096),
            'KORD': (41.9786, -87.9048),
            'KLAX': (33.9425, -118.4081),
            'KDFW': (32.8968, -97.0380),
            'KATL': (33.6407, -84.4277),
            'KMIA': (25.7959, -80.2870),
            'KSEA': (47.4502, -122.3088),
            'KSFO': (37.6213, -122.3790),
            'KPHX': (33.4342, -112.0116),
            'KLAS': (36.0840, -115.1537),
            'KIAH': (29.9844, -95.3414),
            'KDTW': (42.2162, -83.3554),
            'KMSP': (44.8848, -93.2223),
            'KPHL': (39.8729, -75.2437),
            'KBWI': (39.1774, -76.6684),
            'KDCA': (38.8512, -77.0402),
            'KIAD': (38.9531, -77.4565),
            'EDDF': (50.0379, 8.5622),  # Frankfurt
            'EGLL': (51.4700, -0.4543),  # London Heathrow
            'LFPG': (49.0097, 2.5479),   # Paris CDG
            'EHAM': (52.3105, 4.7683),  # Amsterdam
            'LEMD': (40.4983, -3.5676),  # Madrid
            'LIRF': (41.8045, 12.2509), # Rome Fiumicino
        }
    
    def get_coordinates(self, icao_code: str) -> Tuple[float, float]:
        """Get airport coordinates by ICAO code"""
        return self.airports.get(icao_code, (0, 0))
    
    def is_valid_airport(self, icao_code: str) -> bool:
        """Check if airport code exists in database"""
        return icao_code in self.airports


class FlightProcessor:
    """Processes flight data and calculates positions"""
    
    def __init__(self, airport_db: AirportDatabase):
        self.airport_db = airport_db
        self.flights: List[FlightPosition] = []
    
    def load_data_json(self, filename: str = 'data.json') -> Dict:
        """Load and parse the data.json file"""
        try:
            with open(filename, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: {filename} not found")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {}
    
    def haversine_offset(self, lat: float, lng: float, distance_m: float, bearing_degrees: float = 0) -> Tuple[float, float]:
        """Convert distance/bearing from airport to lat/lng coordinates"""
        R = 6371000  # Earth radius in meters
        bearing = math.radians(bearing_degrees)
        
        lat1 = math.radians(lat)
        lng1 = math.radians(lng)
        
        lat2 = math.asin(math.sin(lat1) * math.cos(distance_m/R) + 
                         math.cos(lat1) * math.sin(distance_m/R) * math.cos(bearing))
        lng2 = lng1 + math.atan2(math.sin(bearing) * math.sin(distance_m/R) * math.cos(lat1),
                                 math.cos(distance_m/R) - math.sin(lat1) * math.sin(lat2))
        
        return math.degrees(lat2), math.degrees(lng2)
    
    def estimate_aircraft_type(self, callsign: str) -> str:
        """Estimate aircraft type from callsign"""
        callsign_clean = callsign.strip().upper()
        
        # Major airline patterns
        if callsign_clean.startswith(('DAL', 'DL')):
            return 'B737'  # Delta primarily uses Boeing
        elif callsign_clean.startswith(('UAL', 'UA')):
            return 'B737'  # United primarily uses Boeing
        elif callsign_clean.startswith(('AAL', 'AA')):
            return 'B737'  # American primarily uses Boeing
        elif callsign_clean.startswith(('JBU', 'B6')):
            return 'A320'  # JetBlue primarily uses Airbus
        elif callsign_clean.startswith(('SWA', 'WN')):
            return 'B737'  # Southwest uses Boeing
        elif callsign_clean.startswith(('DLH', 'LH')):
            return 'A320'  # Lufthansa uses Airbus
        elif callsign_clean.startswith(('AFR', 'AF')):
            return 'A320'  # Air France uses Airbus
        elif callsign_clean.startswith(('BAW', 'BA')):
            return 'A320'  # British Airways uses Airbus
        else:
            return 'B737'  # Default to most common type
    
    def process_departure_data(self, departure: Dict) -> List[FlightPosition]:
        """Process a single departure record into flight positions"""
        positions = []
        
        # Extract basic info
        callsign = departure.get('callsign', 'UNKNOWN').strip()
        icao24 = departure.get('icao24', 'unknown')
        airport_code = departure.get('estDepartureAirport', '')
        
        # Get airport coordinates
        airport_lat, airport_lng = self.airport_db.get_coordinates(airport_code)
        if airport_lat == 0 and airport_lng == 0:
            return positions  # Skip invalid airports
        
        # Extract distances and timestamps
        horiz_distance = departure.get('estDepartureAirportHorizDistance', 0)
        vert_distance = departure.get('estDepartureAirportVertDistance', 0)
        first_seen = departure.get('firstSeen', 0)
        last_seen = departure.get('lastSeen', 0)
        
        # Convert timestamps
        first_time = datetime.fromtimestamp(first_seen) if first_seen else datetime.now()
        last_time = datetime.fromtimestamp(last_seen) if last_seen else datetime.now()
        
        # Calculate flight positions along the path
        # Assume aircraft moves away from airport over time
        time_diff = (last_time - first_time).total_seconds()
        if time_diff <= 0:
            time_diff = 1
        
        # Generate positions at different times
        num_positions = min(10, max(3, int(time_diff / 60)))  # 1 position per minute, max 10
        
        for i in range(num_positions):
            # Calculate position at this time
            progress = i / max(1, num_positions - 1)
            
            # Distance increases over time (aircraft departing)
            current_distance = horiz_distance + (progress * 1000)  # Add up to 1km
            
            # Random bearing for departure (0-360 degrees)
            bearing = np.random.uniform(0, 360)
            
            # Calculate position
            lat, lng = self.haversine_offset(airport_lat, airport_lng, current_distance, bearing)
            
            # Calculate altitude (increases over time)
            altitude = vert_distance + (progress * 1000)  # Climb rate
            
            # Calculate timestamp
            time_offset = timedelta(seconds=progress * time_diff)
            timestamp = first_time + time_offset
            
            # Create flight position
            position = FlightPosition(
                callsign=callsign,
                icao24=icao24,
                lat=lat,
                lng=lng,
                altitude=altitude,
                timestamp=timestamp,
                aircraft_type=self.estimate_aircraft_type(callsign),
                airport=airport_code
            )
            
            positions.append(position)
        
        return positions
    
    def process_data_json(self, filename: str = 'data.json') -> List[FlightPosition]:
        """Process the entire data.json file"""
        data = self.load_data_json(filename)
        if not data:
            return []
        
        departures = data.get('departures', [])
        all_positions = []
        
        print(f"Processing {len(departures)} departure records...")
        
        for i, departure in enumerate(departures):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(departures)} records...")
            
            positions = self.process_departure_data(departure)
            all_positions.extend(positions)
        
        self.flights = all_positions
        print(f"Generated {len(all_positions)} flight positions")
        return all_positions
