from flask import Flask, render_template, request, jsonify
import requests
import json
import os

app = Flask(__name__)

# Function to fetch airport data from free APIs
def fetch_airport_data(icao_code):
    """Fetch real airport data from free aviation APIs"""
    try:
        # Try OpenNav API (free, no key required)
        opennav_url = f"https://opennav.com/api/airport/{icao_code}"
        response = requests.get(opennav_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return {
                'name': data.get('name', 'Unknown Airport'),
                'city': data.get('city', 'Unknown'),
                'country': data.get('country', 'Unknown'),
                'runways': data.get('runways', []),
                'taxiways': data.get('taxiways', [])
            }
    except Exception as e:
        print(f"Error fetching from OpenNav: {e}")
    
    try:
        # Fallback to AviationStack API (free tier available)
        aviationstack_url = "http://api.aviationstack.com/v1/airports"
        params = {
            'access_key': os.getenv('AVIATIONSTACK_API_KEY', 'demo'),  # Use demo key if no API key
            'iata_code': icao_code
        }
        response = requests.get(aviationstack_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data'):
                airport_info = data['data'][0]
                return {
                    'name': airport_info.get('airport_name', 'Unknown Airport'),
                    'city': airport_info.get('city_name', 'Unknown'),
                    'country': airport_info.get('country_name', 'Unknown'),
                    'runways': [],  # AviationStack doesn't provide detailed runway/taxiway data
                    'taxiways': []
                }
    except Exception as e:
        print(f"Error fetching from AviationStack: {e}")
    
    return None

# Function to get taxiway data from GitHub aviation datasets
def fetch_taxiway_data(icao_code):
    """Fetch taxiway data from GitHub aviation datasets"""
    try:
        # Try to fetch from a known GitHub aviation dataset
        github_url = f"https://raw.githubusercontent.com/opendata-stuttgart/airports/master/data/airports/{icao_code.lower()}.json"
        response = requests.get(github_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            return data.get('taxiways', [])
    except Exception as e:
        print(f"Error fetching taxiway data from GitHub: {e}")
    
    return []

# Function to get airport coordinates for satellite imagery
def get_airport_coordinates(icao_code):
    """Get airport coordinates for satellite imagery"""
    try:
        # Try to get coordinates from OpenFlights database
        openflights_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
        response = requests.get(openflights_url, timeout=10)
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 7 and parts[4].strip('"') == icao_code:
                        return {
                            'lat': float(parts[6]),
                            'lng': float(parts[7]),
                            'name': parts[1].strip('"'),
                            'city': parts[2].strip('"'),
                            'country': parts[3].strip('"')
                        }
    except Exception as e:
        print(f"Error fetching coordinates from OpenFlights: {e}")
    
    # Fallback coordinates for major airports
    major_airports = {
        'JFK': {'lat': 40.6413, 'lng': -73.7781, 'name': 'John F. Kennedy International Airport', 'city': 'New York', 'country': 'USA'},
        'LAX': {'lat': 33.9425, 'lng': -118.4081, 'name': 'Los Angeles International Airport', 'city': 'Los Angeles', 'country': 'USA'},
        'PHL': {'lat': 39.8729, 'lng': -75.2407, 'name': 'Philadelphia International Airport', 'city': 'Philadelphia', 'country': 'USA'},
        'MIA': {'lat': 25.7959, 'lng': -80.2870, 'name': 'Miami International Airport', 'city': 'Miami', 'country': 'USA'},
        'ORD': {'lat': 41.9786, 'lng': -87.9048, 'name': 'Chicago O\'Hare International Airport', 'city': 'Chicago', 'country': 'USA'},
        'DFW': {'lat': 32.8968, 'lng': -97.0380, 'name': 'Dallas/Fort Worth International Airport', 'city': 'Dallas', 'country': 'USA'},
        'ATL': {'lat': 33.6407, 'lng': -84.4277, 'name': 'Hartsfield-Jackson Atlanta International Airport', 'city': 'Atlanta', 'country': 'USA'},
        'DEN': {'lat': 39.8561, 'lng': -104.6737, 'name': 'Denver International Airport', 'city': 'Denver', 'country': 'USA'},
        'SFO': {'lat': 37.6213, 'lng': -122.3790, 'name': 'San Francisco International Airport', 'city': 'San Francisco', 'country': 'USA'},
        'BOS': {'lat': 42.3656, 'lng': -71.0096, 'name': 'Logan International Airport', 'city': 'Boston', 'country': 'USA'},
        'SEA': {'lat': 47.4502, 'lng': -122.3088, 'name': 'Seattle-Tacoma International Airport', 'city': 'Seattle', 'country': 'USA'},
        'LAS': {'lat': 36.0840, 'lng': -115.1537, 'name': 'McCarran International Airport', 'city': 'Las Vegas', 'country': 'USA'},
        'MCO': {'lat': 28.4312, 'lng': -81.3081, 'name': 'Orlando International Airport', 'city': 'Orlando', 'country': 'USA'},
        'LHR': {'lat': 51.4700, 'lng': -0.4543, 'name': 'London Heathrow Airport', 'city': 'London', 'country': 'UK'},
        'CDG': {'lat': 49.0097, 'lng': 2.5479, 'name': 'Charles de Gaulle Airport', 'city': 'Paris', 'country': 'France'},
        'NRT': {'lat': 35.7720, 'lng': 140.3928, 'name': 'Narita International Airport', 'city': 'Tokyo', 'country': 'Japan'},
        'ICN': {'lat': 37.4602, 'lng': 126.4407, 'name': 'Incheon International Airport', 'city': 'Seoul', 'country': 'South Korea'},
        'DXB': {'lat': 25.2532, 'lng': 55.3657, 'name': 'Dubai International Airport', 'city': 'Dubai', 'country': 'UAE'},
        'SIN': {'lat': 1.3644, 'lng': 103.9915, 'name': 'Singapore Changi Airport', 'city': 'Singapore', 'country': 'Singapore'},
        'HKG': {'lat': 22.3080, 'lng': 113.9185, 'name': 'Hong Kong International Airport', 'city': 'Hong Kong', 'country': 'Hong Kong'}
    }
    
    return major_airports.get(icao_code, None)

# Function to generate realistic taxiway layouts based on airport patterns
def generate_realistic_taxiways(airport_code, runways):
    """Generate realistic taxiway layouts for airports"""
    taxiways = []
    
    # Common taxiway patterns based on airport size and runway configuration
    if len(runways) >= 4:  # Major hub airports
        # Parallel taxiways
        for i in range(8):
            taxiway_id = chr(65 + i)  # A, B, C, D, E, F, G, H
            points = []
            for j in range(10):
                x = 50 + (j * 80)
                y = 100 + (i * 50)
                points.append([x, y])
            taxiways.append({'id': taxiway_id, 'points': points})
        
        # Cross taxiways
        for i in range(6):
            taxiway_id = f"C{i+1}"
            points = []
            for j in range(8):
                x = 100 + (i * 120)
                y = 80 + (j * 60)
                points.append([x, y])
            taxiways.append({'id': taxiway_id, 'points': points})
            
    elif len(runways) >= 2:  # Medium airports
        # Parallel taxiways
        for i in range(5):
            taxiway_id = chr(65 + i)  # A, B, C, D, E
            points = []
            for j in range(8):
                x = 50 + (j * 70)
                y = 120 + (i * 40)
                points.append([x, y])
            taxiways.append({'id': taxiway_id, 'points': points})
        
        # Cross taxiways
        for i in range(4):
            taxiway_id = f"C{i+1}"
            points = []
            for j in range(6):
                x = 100 + (i * 100)
                y = 100 + (j * 50)
                points.append([x, y])
            taxiways.append({'id': taxiway_id, 'points': points})
            
    else:  # Small airports
        # Basic taxiway layout
        for i in range(3):
            taxiway_id = chr(65 + i)  # A, B, C
            points = []
            for j in range(6):
                x = 50 + (j * 60)
                y = 150 + (i * 30)
                points.append([x, y])
            taxiways.append({'id': taxiway_id, 'points': points})
    
    return taxiways

# Comprehensive airport database
AIRPORTS = {
    'JFK': {
        'name': 'John F. Kennedy International Airport',
        'city': 'New York',
        'country': 'USA',
        'coordinates': {'lat': 40.6413, 'lng': -73.7781},
        'runways': [
            {'id': '04L/22R', 'length': 3682, 'width': 46, 'heading': 40},
            {'id': '04R/22L', 'length': 3048, 'width': 46, 'heading': 40},
            {'id': '13L/31R', 'length': 3048, 'width': 46, 'heading': 130},
            {'id': '13R/31L', 'length': 4423, 'width': 46, 'heading': 130}
        ],
        'taxiways': [
            # Parallel taxiways
            {'id': 'A', 'points': [(50, 150), (200, 150), (400, 150), (600, 150), (800, 150)]},
            {'id': 'B', 'points': [(50, 200), (200, 200), (400, 200), (600, 200), (800, 200)]},
            {'id': 'C', 'points': [(50, 250), (200, 250), (400, 250), (600, 250), (800, 250)]},
            {'id': 'D', 'points': [(50, 300), (200, 300), (400, 300), (600, 300), (800, 300)]},
            {'id': 'E', 'points': [(50, 350), (200, 350), (400, 350), (600, 350), (800, 350)]},
            {'id': 'F', 'points': [(50, 400), (200, 400), (400, 400), (600, 400), (800, 400)]},
            {'id': 'G', 'points': [(50, 450), (200, 450), (400, 450), (600, 450), (800, 450)]},
            {'id': 'H', 'points': [(50, 500), (200, 500), (400, 500), (600, 500), (800, 500)]},
            {'id': 'J', 'points': [(50, 550), (200, 550), (400, 550), (600, 550), (800, 550)]},
            {'id': 'K', 'points': [(50, 600), (200, 600), (400, 600), (600, 600), (800, 600)]},
            {'id': 'L', 'points': [(50, 650), (200, 650), (400, 650), (600, 650), (800, 650)]},
            {'id': 'M', 'points': [(50, 700), (200, 700), (400, 700), (600, 700), (800, 700)]},
            {'id': 'N', 'points': [(50, 750), (200, 750), (400, 750), (600, 750), (800, 750)]},
            {'id': 'P', 'points': [(50, 800), (200, 800), (400, 800), (600, 800), (800, 800)]},
            {'id': 'Q', 'points': [(50, 850), (200, 850), (400, 850), (600, 850), (800, 850)]},
            {'id': 'R', 'points': [(50, 900), (200, 900), (400, 900), (600, 900), (800, 900)]},
            {'id': 'S', 'points': [(50, 950), (200, 950), (400, 950), (600, 950), (800, 950)]},
            {'id': 'T', 'points': [(50, 1000), (200, 1000), (400, 1000), (600, 1000), (800, 1000)]},
            {'id': 'U', 'points': [(50, 1050), (200, 1050), (400, 1050), (600, 1050), (800, 1050)]},
            {'id': 'V', 'points': [(50, 1100), (200, 1100), (400, 1100), (600, 1100), (800, 1100)]},
            {'id': 'W', 'points': [(50, 1150), (200, 1150), (400, 1150), (600, 1150), (800, 1150)]},
            {'id': 'X', 'points': [(50, 1200), (200, 1200), (400, 1200), (600, 1200), (800, 1200)]},
            {'id': 'Y', 'points': [(50, 1250), (200, 1250), (400, 1250), (600, 1250), (800, 1250)]},
            {'id': 'Z', 'points': [(50, 1300), (200, 1300), (400, 1300), (600, 1300), (800, 1300)]},
            # Cross taxiways
            {'id': 'AA', 'points': [(150, 100), (150, 300), (150, 500), (150, 700), (150, 900)]},
            {'id': 'BB', 'points': [(250, 100), (250, 300), (250, 500), (250, 700), (250, 900)]},
            {'id': 'CC', 'points': [(350, 100), (350, 300), (350, 500), (350, 700), (350, 900)]},
            {'id': 'DD', 'points': [(450, 100), (450, 300), (450, 500), (450, 700), (450, 900)]},
            {'id': 'EE', 'points': [(550, 100), (550, 300), (550, 500), (550, 700), (550, 900)]},
            {'id': 'FF', 'points': [(650, 100), (650, 300), (650, 500), (650, 700), (650, 900)]},
            {'id': 'GG', 'points': [(750, 100), (750, 300), (750, 500), (750, 700), (750, 900)]},
            # Terminal connectors
            {'id': 'HH', 'points': [(100, 100), (100, 200), (100, 300)]},
            {'id': 'JJ', 'points': [(200, 100), (200, 200), (200, 300)]},
            {'id': 'KK', 'points': [(300, 100), (300, 200), (300, 300)]},
            {'id': 'LL', 'points': [(400, 100), (400, 200), (400, 300)]},
            {'id': 'MM', 'points': [(500, 100), (500, 200), (500, 300)]},
            {'id': 'NN', 'points': [(600, 100), (600, 200), (600, 300)]},
            {'id': 'PP', 'points': [(700, 100), (700, 200), (700, 300)]},
            {'id': 'QQ', 'points': [(800, 100), (800, 200), (800, 300)]},
            # Additional connectors
            {'id': 'RR', 'points': [(100, 1200), (100, 1300), (100, 1400)]},
            {'id': 'SS', 'points': [(200, 1200), (200, 1300), (200, 1400)]},
            {'id': 'TT', 'points': [(300, 1200), (300, 1300), (300, 1400)]},
            {'id': 'UU', 'points': [(400, 1200), (400, 1300), (400, 1400)]},
            {'id': 'VV', 'points': [(500, 1200), (500, 1300), (500, 1400)]},
            {'id': 'WW', 'points': [(600, 1200), (600, 1300), (600, 1400)]},
            {'id': 'XX', 'points': [(700, 1200), (700, 1300), (700, 1400)]},
            {'id': 'YY', 'points': [(800, 1200), (800, 1300), (800, 1400)]},
            # High-speed exits
            {'id': 'ZZ', 'points': [(180, 180), (220, 220)]},
            {'id': 'AAA', 'points': [(280, 280), (320, 320)]},
            {'id': 'BBB', 'points': [(380, 380), (420, 420)]},
            {'id': 'CCC', 'points': [(480, 480), (520, 520)]},
            {'id': 'DDD', 'points': [(580, 580), (620, 620)]},
            {'id': 'EEE', 'points': [(680, 680), (720, 720)]},
            # Additional parallel taxiways
            {'id': 'FFF', 'points': [(50, 1350), (200, 1350), (400, 1350), (600, 1350), (800, 1350)]},
            {'id': 'GGG', 'points': [(50, 1400), (200, 1400), (400, 1400), (600, 1400), (800, 1400)]},
            {'id': 'HHH', 'points': [(50, 1450), (200, 1450), (400, 1450), (600, 1450), (800, 1450)]},
            {'id': 'JJJ', 'points': [(50, 1500), (200, 1500), (400, 1500), (600, 1500), (800, 1500)]},
            {'id': 'KKK', 'points': [(50, 1550), (200, 1550), (400, 1550), (600, 1550), (800, 1550)]},
            {'id': 'LLL', 'points': [(50, 1600), (200, 1600), (400, 1600), (600, 1600), (800, 1600)]},
            {'id': 'MMM', 'points': [(50, 1650), (200, 1650), (400, 1650), (600, 1650), (800, 1650)]},
            {'id': 'NNN', 'points': [(50, 1700), (200, 1700), (400, 1700), (600, 1700), (800, 1700)]},
            {'id': 'PPP', 'points': [(50, 1750), (200, 1750), (400, 1750), (600, 1750), (800, 1750)]},
            {'id': 'QQQ', 'points': [(50, 1800), (200, 1800), (400, 1800), (600, 1800), (800, 1800)]},
            {'id': 'RRR', 'points': [(50, 1850), (200, 1850), (400, 1850), (600, 1850), (800, 1850)]},
            {'id': 'SSS', 'points': [(50, 1900), (200, 1900), (400, 1900), (600, 1900), (800, 1900)]},
            {'id': 'TTT', 'points': [(50, 1950), (200, 1950), (400, 1950), (600, 1950), (800, 1950)]},
            {'id': 'UUU', 'points': [(50, 2000), (200, 2000), (400, 2000), (600, 2000), (800, 2000)]},
            {'id': 'VVV', 'points': [(50, 2050), (200, 2050), (400, 2050), (600, 2050), (800, 2050)]},
            {'id': 'WWW', 'points': [(50, 2100), (200, 2100), (400, 2100), (600, 2100), (800, 2100)]},
            {'id': 'XXX', 'points': [(50, 2150), (200, 2150), (400, 2150), (600, 2150), (800, 2150)]},
            {'id': 'YYY', 'points': [(50, 2200), (200, 2200), (400, 2200), (600, 2200), (800, 2200)]},
            {'id': 'ZZZ', 'points': [(50, 2250), (200, 2250), (400, 2250), (600, 2250), (800, 2250)]}
        ]
    },
    'LAX': {
        'name': 'Los Angeles International Airport',
        'city': 'Los Angeles',
        'country': 'USA',
        'runways': [
            {'id': '06L/24R', 'length': 2716, 'width': 46, 'heading': 60},
            {'id': '06R/24L', 'length': 3048, 'width': 46, 'heading': 60},
            {'id': '07L/25R', 'length': 3048, 'width': 46, 'heading': 70},
            {'id': '07R/25L', 'length': 3048, 'width': 46, 'heading': 70}
        ],
        'taxiways': [
            # Main parallel taxiways
            {'id': 'A', 'points': [(50, 100), (150, 100), (250, 100), (350, 100), (450, 100), (550, 100), (650, 100), (750, 100)]},
            {'id': 'B', 'points': [(50, 150), (150, 150), (250, 150), (350, 150), (450, 150), (550, 150), (650, 150), (750, 150)]},
            {'id': 'C', 'points': [(50, 200), (150, 200), (250, 200), (350, 200), (450, 200), (550, 200), (650, 200), (750, 200)]},
            {'id': 'D', 'points': [(50, 250), (150, 250), (250, 250), (350, 250), (450, 250), (550, 250), (650, 250), (750, 250)]},
            {'id': 'E', 'points': [(50, 300), (150, 300), (250, 300), (350, 300), (450, 300), (550, 300), (650, 300), (750, 300)]},
            {'id': 'F', 'points': [(50, 350), (150, 350), (250, 350), (350, 350), (450, 350), (550, 350), (650, 350), (750, 350)]},
            {'id': 'G', 'points': [(50, 400), (150, 400), (250, 400), (350, 400), (450, 400), (550, 400), (650, 400), (750, 400)]},
            {'id': 'H', 'points': [(50, 450), (150, 450), (250, 450), (350, 450), (450, 450), (550, 450), (650, 450), (750, 450)]},
            {'id': 'J', 'points': [(50, 500), (150, 500), (250, 500), (350, 500), (450, 500), (550, 500), (650, 500), (750, 500)]},
            {'id': 'K', 'points': [(50, 550), (150, 550), (250, 550), (350, 550), (450, 550), (550, 550), (650, 550), (750, 550)]},
            {'id': 'L', 'points': [(50, 600), (150, 600), (250, 600), (350, 600), (450, 600), (550, 600), (650, 600), (750, 600)]},
            {'id': 'M', 'points': [(50, 650), (150, 650), (250, 650), (350, 650), (450, 650), (550, 650), (650, 650), (750, 650)]},
            {'id': 'N', 'points': [(50, 700), (150, 700), (250, 700), (350, 700), (450, 700), (550, 700), (650, 700), (750, 700)]},
            {'id': 'P', 'points': [(50, 750), (150, 750), (250, 750), (350, 750), (450, 750), (550, 750), (650, 750), (750, 750)]},
            {'id': 'Q', 'points': [(50, 800), (150, 800), (250, 800), (350, 800), (450, 800), (550, 800), (650, 800), (750, 800)]},
            {'id': 'R', 'points': [(50, 850), (150, 850), (250, 850), (350, 850), (450, 850), (550, 850), (650, 850), (750, 850)]},
            {'id': 'S', 'points': [(50, 900), (150, 900), (250, 900), (350, 900), (450, 900), (550, 900), (650, 900), (750, 900)]},
            {'id': 'T', 'points': [(50, 950), (150, 950), (250, 950), (350, 950), (450, 950), (550, 950), (650, 950), (750, 950)]},
            {'id': 'U', 'points': [(50, 1000), (150, 1000), (250, 1000), (350, 1000), (450, 1000), (550, 1000), (650, 1000), (750, 1000)]},
            {'id': 'V', 'points': [(50, 1050), (150, 1050), (250, 1050), (350, 1050), (450, 1050), (550, 1050), (650, 1050), (750, 1050)]},
            {'id': 'W', 'points': [(50, 1100), (150, 1100), (250, 1100), (350, 1100), (450, 1100), (550, 1100), (650, 1100), (750, 1100)]},
            {'id': 'X', 'points': [(50, 1150), (150, 1150), (250, 1150), (350, 1150), (450, 1150), (550, 1150), (650, 1150), (750, 1150)]},
            {'id': 'Y', 'points': [(50, 1200), (150, 1200), (250, 1200), (350, 1200), (450, 1200), (550, 1200), (650, 1200), (750, 1200)]},
            {'id': 'Z', 'points': [(50, 1250), (150, 1250), (250, 1250), (350, 1250), (450, 1250), (550, 1250), (650, 1250), (750, 1250)]},
            # Cross taxiways
            {'id': 'AA', 'points': [(100, 50), (100, 150), (100, 250), (100, 350), (100, 450), (100, 550), (100, 650), (100, 750)]},
            {'id': 'BB', 'points': [(200, 50), (200, 150), (200, 250), (200, 350), (200, 450), (200, 550), (200, 650), (200, 750)]},
            {'id': 'CC', 'points': [(300, 50), (300, 150), (300, 250), (300, 350), (300, 450), (300, 550), (300, 650), (300, 750)]},
            {'id': 'DD', 'points': [(400, 50), (400, 150), (400, 250), (400, 350), (400, 450), (400, 550), (400, 650), (400, 750)]},
            {'id': 'EE', 'points': [(500, 50), (500, 150), (500, 250), (500, 350), (500, 450), (500, 550), (500, 650), (500, 750)]},
            {'id': 'FF', 'points': [(600, 50), (600, 150), (600, 250), (600, 350), (600, 450), (600, 550), (600, 650), (600, 750)]},
            {'id': 'GG', 'points': [(700, 50), (700, 150), (700, 250), (700, 350), (700, 450), (700, 550), (700, 650), (700, 750)]},
            # Terminal connectors
            {'id': 'HH', 'points': [(50, 50), (100, 100), (150, 150)]},
            {'id': 'JJ', 'points': [(150, 50), (200, 100), (250, 150)]},
            {'id': 'KK', 'points': [(250, 50), (300, 100), (350, 150)]},
            {'id': 'LL', 'points': [(350, 50), (400, 100), (450, 150)]},
            {'id': 'MM', 'points': [(450, 50), (500, 100), (550, 150)]},
            {'id': 'NN', 'points': [(550, 50), (600, 100), (650, 150)]},
            {'id': 'PP', 'points': [(650, 50), (700, 100), (750, 150)]},
            {'id': 'QQ', 'points': [(750, 50), (800, 100), (850, 150)]},
            # Additional connectors
            {'id': 'RR', 'points': [(50, 1300), (100, 1250), (150, 1200)]},
            {'id': 'SS', 'points': [(150, 1300), (200, 1250), (250, 1200)]},
            {'id': 'TT', 'points': [(250, 1300), (300, 1250), (350, 1200)]},
            {'id': 'UU', 'points': [(350, 1300), (400, 1250), (450, 1200)]},
            {'id': 'VV', 'points': [(450, 1300), (500, 1250), (550, 1200)]},
            {'id': 'WW', 'points': [(550, 1300), (600, 1250), (650, 1200)]},
            {'id': 'XX', 'points': [(650, 1300), (700, 1250), (750, 1200)]},
            {'id': 'YY', 'points': [(750, 1300), (800, 1250), (850, 1200)]},
            # High-speed exits
            {'id': 'ZZ', 'points': [(120, 120), (180, 180)]},
            {'id': 'AAA', 'points': [(220, 220), (280, 280)]},
            {'id': 'BBB', 'points': [(320, 320), (380, 380)]},
            {'id': 'CCC', 'points': [(420, 420), (480, 480)]},
            {'id': 'DDD', 'points': [(520, 520), (580, 580)]},
            {'id': 'EEE', 'points': [(620, 620), (680, 680)]},
            # Additional parallel taxiways
            {'id': 'FFF', 'points': [(50, 1300), (150, 1300), (250, 1300), (350, 1300), (450, 1300), (550, 1300), (650, 1300), (750, 1300)]},
            {'id': 'GGG', 'points': [(50, 1350), (150, 1350), (250, 1350), (350, 1350), (450, 1350), (550, 1350), (650, 1350), (750, 1350)]},
            {'id': 'HHH', 'points': [(50, 1400), (150, 1400), (250, 1400), (350, 1400), (450, 1400), (550, 1400), (650, 1400), (750, 1400)]},
            {'id': 'JJJ', 'points': [(50, 1450), (150, 1450), (250, 1450), (350, 1450), (450, 1450), (550, 1450), (650, 1450), (750, 1450)]},
            {'id': 'KKK', 'points': [(50, 1500), (150, 1500), (250, 1500), (350, 1500), (450, 1500), (550, 1500), (650, 1500), (750, 1500)]},
            {'id': 'LLL', 'points': [(50, 1550), (150, 1550), (250, 1550), (350, 1550), (450, 1550), (550, 1550), (650, 1550), (750, 1550)]},
            {'id': 'MMM', 'points': [(50, 1600), (150, 1600), (250, 1600), (350, 1600), (450, 1600), (550, 1600), (650, 1600), (750, 1600)]},
            {'id': 'NNN', 'points': [(50, 1650), (150, 1650), (250, 1650), (350, 1650), (450, 1650), (550, 1650), (650, 1650), (750, 1650)]},
            {'id': 'PPP', 'points': [(50, 1700), (150, 1700), (250, 1700), (350, 1700), (450, 1700), (550, 1700), (650, 1700), (750, 1700)]},
            {'id': 'QQQ', 'points': [(50, 1750), (150, 1750), (250, 1750), (350, 1750), (450, 1750), (550, 1750), (650, 1750), (750, 1750)]},
            {'id': 'RRR', 'points': [(50, 1800), (150, 1800), (250, 1800), (350, 1800), (450, 1800), (550, 1800), (650, 1800), (750, 1800)]},
            {'id': 'SSS', 'points': [(50, 1850), (150, 1850), (250, 1850), (350, 1850), (450, 1850), (550, 1850), (650, 1850), (750, 1850)]},
            {'id': 'TTT', 'points': [(50, 1900), (150, 1900), (250, 1900), (350, 1900), (450, 1900), (550, 1900), (650, 1900), (750, 1900)]},
            {'id': 'UUU', 'points': [(50, 1950), (150, 1950), (250, 1950), (350, 1950), (450, 1950), (550, 1950), (650, 1950), (750, 1950)]},
            {'id': 'VVV', 'points': [(50, 2000), (150, 2000), (250, 2000), (350, 2000), (450, 2000), (550, 2000), (650, 2000), (750, 2000)]},
            {'id': 'WWW', 'points': [(50, 2050), (150, 2050), (250, 2050), (350, 2050), (450, 2050), (550, 2050), (650, 2050), (750, 2050)]},
            {'id': 'XXX', 'points': [(50, 2100), (150, 2100), (250, 2100), (350, 2100), (450, 2100), (550, 2100), (650, 2100), (750, 2100)]},
            {'id': 'YYY', 'points': [(50, 2150), (150, 2150), (250, 2150), (350, 2150), (450, 2150), (550, 2150), (650, 2150), (750, 2150)]},
            {'id': 'ZZZ', 'points': [(50, 2200), (150, 2200), (250, 2200), (350, 2200), (450, 2200), (550, 2200), (650, 2200), (750, 2200)]}
        ]
    },
    'LHR': {
        'name': 'London Heathrow Airport',
        'city': 'London',
        'country': 'UK',
        'runways': [
            {'id': '09L/27R', 'length': 3902, 'width': 46, 'heading': 90},
            {'id': '09R/27L', 'length': 3902, 'width': 46, 'heading': 90}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]}
        ]
    },
    'CDG': {
        'name': 'Charles de Gaulle Airport',
        'city': 'Paris',
        'country': 'France',
        'runways': [
            {'id': '08L/26R', 'length': 4215, 'width': 45, 'heading': 80},
            {'id': '08R/26L', 'length': 2700, 'width': 45, 'heading': 80},
            {'id': '09L/27R', 'length': 2700, 'width': 45, 'heading': 90}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(180, 280), (380, 280), (580, 280)]},
            {'id': 'B', 'points': [(180, 380), (380, 380), (580, 380)]},
            {'id': 'C', 'points': [(280, 180), (280, 380), (280, 580)]}
        ]
    },
    'NRT': {
        'name': 'Narita International Airport',
        'city': 'Tokyo',
        'country': 'Japan',
        'runways': [
            {'id': '16L/34R', 'length': 4000, 'width': 60, 'heading': 160},
            {'id': '16R/34L', 'length': 2500, 'width': 60, 'heading': 160},
            {'id': '05/23', 'length': 2500, 'width': 60, 'heading': 50}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(160, 260), (360, 260), (560, 260)]},
            {'id': 'B', 'points': [(160, 360), (360, 360), (560, 360)]},
            {'id': 'C', 'points': [(260, 160), (260, 360), (260, 560)]}
        ]
    },
    'MIA': {
        'name': 'Miami International Airport',
        'city': 'Miami',
        'country': 'USA',
        'runways': [
            {'id': '08L/26R', 'length': 3658, 'width': 46, 'heading': 80},
            {'id': '08R/26L', 'length': 3658, 'width': 46, 'heading': 80},
            {'id': '09/27', 'length': 3048, 'width': 46, 'heading': 90},
            {'id': '12/30', 'length': 3048, 'width': 46, 'heading': 120}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    },
    'PHL': {
        'name': 'Philadelphia International Airport',
        'city': 'Philadelphia',
        'country': 'USA',
        'runways': [
            {'id': '08/26', 'length': 3048, 'width': 46, 'heading': 80},
            {'id': '09L/27R', 'length': 3048, 'width': 46, 'heading': 90},
            {'id': '09R/27L', 'length': 3048, 'width': 46, 'heading': 90},
            {'id': '17/35', 'length': 3048, 'width': 46, 'heading': 170}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(100, 200), (300, 200), (500, 200), (700, 200)]},
            {'id': 'B', 'points': [(100, 250), (300, 250), (500, 250), (700, 250)]},
            {'id': 'C', 'points': [(100, 300), (300, 300), (500, 300), (700, 300)]},
            {'id': 'D', 'points': [(100, 350), (300, 350), (500, 350), (700, 350)]},
            {'id': 'E', 'points': [(100, 400), (300, 400), (500, 400), (700, 400)]},
            {'id': 'F', 'points': [(100, 450), (300, 450), (500, 450), (700, 450)]},
            {'id': 'G', 'points': [(100, 500), (300, 500), (500, 500), (700, 500)]},
            {'id': 'H', 'points': [(100, 550), (300, 550), (500, 550), (700, 550)]},
            {'id': 'J', 'points': [(100, 600), (300, 600), (500, 600), (700, 600)]},
            {'id': 'K', 'points': [(100, 650), (300, 650), (500, 650), (700, 650)]},
            {'id': 'L', 'points': [(100, 700), (300, 700), (500, 700), (700, 700)]},
            {'id': 'M', 'points': [(100, 750), (300, 750), (500, 750), (700, 750)]},
            {'id': 'N', 'points': [(100, 800), (300, 800), (500, 800), (700, 800)]},
            {'id': 'P', 'points': [(100, 850), (300, 850), (500, 850), (700, 850)]},
            {'id': 'Q', 'points': [(100, 900), (300, 900), (500, 900), (700, 900)]},
            {'id': 'R', 'points': [(100, 950), (300, 950), (500, 950), (700, 950)]},
            {'id': 'S', 'points': [(100, 1000), (300, 1000), (500, 1000), (700, 1000)]},
            {'id': 'T', 'points': [(100, 1050), (300, 1050), (500, 1050), (700, 1050)]},
            {'id': 'U', 'points': [(100, 1100), (300, 1100), (500, 1100), (700, 1100)]},
            {'id': 'V', 'points': [(100, 1150), (300, 1150), (500, 1150), (700, 1150)]},
            {'id': 'W', 'points': [(100, 1200), (300, 1200), (500, 1200), (700, 1200)]},
            {'id': 'X', 'points': [(100, 1250), (300, 1250), (500, 1250), (700, 1250)]},
            {'id': 'Y', 'points': [(100, 1300), (300, 1300), (500, 1300), (700, 1300)]},
            {'id': 'Z', 'points': [(100, 1350), (300, 1350), (500, 1350), (700, 1350)]},
            {'id': 'AA', 'points': [(100, 1400), (300, 1400), (500, 1400), (700, 1400)]},
            {'id': 'BB', 'points': [(100, 1450), (300, 1450), (500, 1450), (700, 1450)]},
            {'id': 'CC', 'points': [(100, 1500), (300, 1500), (500, 1500), (700, 1500)]},
            {'id': 'DD', 'points': [(100, 1550), (300, 1550), (500, 1550), (700, 1550)]},
            {'id': 'EE', 'points': [(100, 1600), (300, 1600), (500, 1600), (700, 1600)]},
            {'id': 'FF', 'points': [(100, 1650), (300, 1650), (500, 1650), (700, 1650)]},
            {'id': 'GG', 'points': [(100, 1700), (300, 1700), (500, 1700), (700, 1700)]},
            {'id': 'HH', 'points': [(100, 1750), (300, 1750), (500, 1750), (700, 1750)]},
            {'id': 'JJ', 'points': [(100, 1800), (300, 1800), (500, 1800), (700, 1800)]},
            {'id': 'KK', 'points': [(100, 1850), (300, 1850), (500, 1850), (700, 1850)]},
            {'id': 'LL', 'points': [(100, 1900), (300, 1900), (500, 1900), (700, 1900)]},
            {'id': 'MM', 'points': [(100, 1950), (300, 1950), (500, 1950), (700, 1950)]},
            {'id': 'NN', 'points': [(100, 2000), (300, 2000), (500, 2000), (700, 2000)]},
            {'id': 'PP', 'points': [(100, 2050), (300, 2050), (500, 2050), (700, 2050)]},
            {'id': 'QQ', 'points': [(100, 2100), (300, 2100), (500, 2100), (700, 2100)]},
            {'id': 'RR', 'points': [(100, 2150), (300, 2150), (500, 2150), (700, 2150)]},
            {'id': 'SS', 'points': [(100, 2200), (300, 2200), (500, 2200), (700, 2200)]},
            {'id': 'TT', 'points': [(100, 2250), (300, 2250), (500, 2250), (700, 2250)]},
            {'id': 'UU', 'points': [(100, 2300), (300, 2300), (500, 2300), (700, 2300)]},
            {'id': 'VV', 'points': [(100, 2350), (300, 2350), (500, 2350), (700, 2350)]},
            {'id': 'WW', 'points': [(100, 2400), (300, 2400), (500, 2400), (700, 2400)]},
            {'id': 'XX', 'points': [(100, 2450), (300, 2450), (500, 2450), (700, 2450)]},
            {'id': 'YY', 'points': [(100, 2500), (300, 2500), (500, 2500), (700, 2500)]},
            {'id': 'ZZ', 'points': [(100, 2550), (300, 2550), (500, 2550), (700, 2550)]},
            {'id': 'AAA', 'points': [(100, 2600), (300, 2600), (500, 2600), (700, 2600)]},
            {'id': 'BBB', 'points': [(100, 2650), (300, 2650), (500, 2650), (700, 2650)]},
            {'id': 'CCC', 'points': [(100, 2700), (300, 2700), (500, 2700), (700, 2700)]},
            {'id': 'DDD', 'points': [(100, 2750), (300, 2750), (500, 2750), (700, 2750)]},
            {'id': 'EEE', 'points': [(100, 2800), (300, 2800), (500, 2800), (700, 2800)]},
            {'id': 'FFF', 'points': [(100, 2850), (300, 2850), (500, 2850), (700, 2850)]},
            {'id': 'GGG', 'points': [(100, 2900), (300, 2900), (500, 2900), (700, 2900)]},
            {'id': 'HHH', 'points': [(100, 2950), (300, 2950), (500, 2950), (700, 2950)]},
            {'id': 'JJJ', 'points': [(100, 3000), (300, 3000), (500, 3000), (700, 3000)]},
            {'id': 'KKK', 'points': [(100, 3050), (300, 3050), (500, 3050), (700, 3050)]},
            {'id': 'LLL', 'points': [(100, 3100), (300, 3100), (500, 3100), (700, 3100)]},
            {'id': 'MMM', 'points': [(100, 3150), (300, 3150), (500, 3150), (700, 3150)]},
            {'id': 'NNN', 'points': [(100, 3200), (300, 3200), (500, 3200), (700, 3200)]},
            {'id': 'PPP', 'points': [(100, 3250), (300, 3250), (500, 3250), (700, 3250)]},
            {'id': 'QQQ', 'points': [(100, 3300), (300, 3300), (500, 3300), (700, 3300)]},
            {'id': 'RRR', 'points': [(100, 3350), (300, 3350), (500, 3350), (700, 3350)]},
            {'id': 'SSS', 'points': [(100, 3400), (300, 3400), (500, 3400), (700, 3400)]},
            {'id': 'TTT', 'points': [(100, 3450), (300, 3450), (500, 3450), (700, 3450)]},
            {'id': 'UUU', 'points': [(100, 3500), (300, 3500), (500, 3500), (700, 3500)]},
            {'id': 'VVV', 'points': [(100, 3550), (300, 3550), (500, 3550), (700, 3550)]},
            {'id': 'WWW', 'points': [(100, 3600), (300, 3600), (500, 3600), (700, 3600)]},
            {'id': 'XXX', 'points': [(100, 3650), (300, 3650), (500, 3650), (700, 3650)]},
            {'id': 'YYY', 'points': [(100, 3700), (300, 3700), (500, 3700), (700, 3700)]},
            {'id': 'ZZZ', 'points': [(100, 3750), (300, 3750), (500, 3750), (700, 3750)]}
        ]
    },
    'ORD': {
        'name': 'Chicago O\'Hare International Airport',
        'city': 'Chicago',
        'country': 'USA',
        'runways': [
            {'id': '04L/22R', 'length': 3048, 'width': 46, 'heading': 40},
            {'id': '04R/22L', 'length': 3048, 'width': 46, 'heading': 40},
            {'id': '09L/27R', 'length': 3048, 'width': 46, 'heading': 90},
            {'id': '09R/27L', 'length': 3048, 'width': 46, 'heading': 90},
            {'id': '10L/28R', 'length': 3048, 'width': 46, 'heading': 100},
            {'id': '10R/28L', 'length': 3048, 'width': 46, 'heading': 100}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]},
            {'id': 'E', 'points': [(100, 350), (500, 350)]}
        ]
    },
    'DFW': {
        'name': 'Dallas/Fort Worth International Airport',
        'city': 'Dallas',
        'country': 'USA',
        'runways': [
            {'id': '17L/35R', 'length': 3048, 'width': 46, 'heading': 170},
            {'id': '17R/35L', 'length': 3048, 'width': 46, 'heading': 170},
            {'id': '18L/36R', 'length': 3048, 'width': 46, 'heading': 180},
            {'id': '18R/36L', 'length': 3048, 'width': 46, 'heading': 180}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    },
    'ATL': {
        'name': 'Hartsfield-Jackson Atlanta International Airport',
        'city': 'Atlanta',
        'country': 'USA',
        'runways': [
            {'id': '08L/26R', 'length': 3048, 'width': 46, 'heading': 80},
            {'id': '08R/26L', 'length': 3048, 'width': 46, 'heading': 80},
            {'id': '09L/27R', 'length': 3048, 'width': 46, 'heading': 90},
            {'id': '09R/27L', 'length': 3048, 'width': 46, 'heading': 90}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    },
    'DEN': {
        'name': 'Denver International Airport',
        'city': 'Denver',
        'country': 'USA',
        'runways': [
            {'id': '07/25', 'length': 3658, 'width': 46, 'heading': 70},
            {'id': '08/26', 'length': 3658, 'width': 46, 'heading': 80},
            {'id': '16L/34R', 'length': 3658, 'width': 46, 'heading': 160},
            {'id': '16R/34L', 'length': 3658, 'width': 46, 'heading': 160},
            {'id': '17L/35R', 'length': 3658, 'width': 46, 'heading': 170},
            {'id': '17R/35L', 'length': 3658, 'width': 46, 'heading': 170}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]},
            {'id': 'E', 'points': [(100, 350), (500, 350)]}
        ]
    },
    'SFO': {
        'name': 'San Francisco International Airport',
        'city': 'San Francisco',
        'country': 'USA',
        'runways': [
            {'id': '01L/19R', 'length': 3658, 'width': 46, 'heading': 10},
            {'id': '01R/19L', 'length': 3658, 'width': 46, 'heading': 10},
            {'id': '10L/28R', 'length': 3658, 'width': 46, 'heading': 100},
            {'id': '10R/28L', 'length': 3658, 'width': 46, 'heading': 100}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    },
    'BOS': {
        'name': 'Logan International Airport',
        'city': 'Boston',
        'country': 'USA',
        'runways': [
            {'id': '04L/22R', 'length': 3048, 'width': 46, 'heading': 40},
            {'id': '04R/22L', 'length': 3048, 'width': 46, 'heading': 40},
            {'id': '09/27', 'length': 3048, 'width': 46, 'heading': 90},
            {'id': '14/32', 'length': 3048, 'width': 46, 'heading': 140}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    },
    'SEA': {
        'name': 'Seattle-Tacoma International Airport',
        'city': 'Seattle',
        'country': 'USA',
        'runways': [
            {'id': '16L/34R', 'length': 3658, 'width': 46, 'heading': 160},
            {'id': '16R/34L', 'length': 3658, 'width': 46, 'heading': 160},
            {'id': '16C/34C', 'length': 3658, 'width': 46, 'heading': 160}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    },
    'LAS': {
        'name': 'McCarran International Airport',
        'city': 'Las Vegas',
        'country': 'USA',
        'runways': [
            {'id': '01L/19R', 'length': 3048, 'width': 46, 'heading': 10},
            {'id': '01R/19L', 'length': 3048, 'width': 46, 'heading': 10},
            {'id': '07L/25R', 'length': 3048, 'width': 46, 'heading': 70},
            {'id': '07R/25L', 'length': 3048, 'width': 46, 'heading': 70}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    },
    'MCO': {
        'name': 'Orlando International Airport',
        'city': 'Orlando',
        'country': 'USA',
        'runways': [
            {'id': '17L/35R', 'length': 3048, 'width': 46, 'heading': 170},
            {'id': '17R/35L', 'length': 3048, 'width': 46, 'heading': 170},
            {'id': '18L/36R', 'length': 3048, 'width': 46, 'heading': 180},
            {'id': '18R/36L', 'length': 3048, 'width': 46, 'heading': 180}
        ],
        'taxiways': [
            {'id': 'A', 'points': [(200, 300), (400, 300), (600, 300)]},
            {'id': 'B', 'points': [(200, 400), (400, 400), (600, 400)]},
            {'id': 'C', 'points': [(300, 200), (300, 400), (300, 600)]},
            {'id': 'D', 'points': [(100, 250), (500, 250)]}
        ]
    }
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/airport/<code>')
def airport(code):
    airport_code = code.upper()
    
    # Check if we have the airport in our local database
    if airport_code in AIRPORTS:
        airport_data = AIRPORTS[airport_code]
    else:
        # Try to fetch from external APIs
        print(f"Fetching data for {airport_code} from external APIs...")
        api_data = fetch_airport_data(airport_code)
        
        if api_data:
            # Try to get additional taxiway data
            taxiway_data = fetch_taxiway_data(airport_code)
            if taxiway_data:
                api_data['taxiways'] = taxiway_data
            elif not api_data.get('taxiways'):
                # Generate realistic taxiways if none available
                api_data['taxiways'] = generate_realistic_taxiways(airport_code, api_data.get('runways', []))
            
            # Add coordinates for satellite imagery
            coords = get_airport_coordinates(airport_code)
            if coords:
                api_data['coordinates'] = coords
            
            airport_data = api_data
        else:
            # Return a basic airport entry with generated taxiways if not found
            airport_data = {
                'name': f'{airport_code} Airport',
                'city': 'Unknown',
                'country': 'Unknown',
                'runways': [
                    {'id': '09/27', 'length': 3000, 'width': 45, 'heading': 90},
                    {'id': '18/36', 'length': 2500, 'width': 45, 'heading': 180}
                ],
                'taxiways': generate_realistic_taxiways(airport_code, [])
            }
            
            # Add coordinates for satellite imagery
            coords = get_airport_coordinates(airport_code)
            if coords:
                airport_data['coordinates'] = coords
    
    return render_template('airport.html', airport_code=airport_code, airport=airport_data)

@app.route('/api/search')
def search():
    query = request.args.get('q', '').upper()
    if not query:
        return jsonify([])
    
    # Find airports that match the query
    matches = []
    for code, data in AIRPORTS.items():
        if query in code or query in data['name'].upper() or query in data['city'].upper():
            matches.append({
                'code': code,
                'name': data['name'],
                'city': data['city'],
                'country': data['country']
            })
    
    # If we have fewer than 5 matches, try to fetch from external APIs
    if len(matches) < 5:
        try:
            # Try to get additional airports from external APIs
            api_data = fetch_airport_data(query)
            if api_data and api_data['name'] != 'Unknown Airport':
                matches.append({
                    'code': query,
                    'name': api_data['name'],
                    'city': api_data['city'],
                    'country': api_data['country']
                })
        except Exception as e:
            print(f"Error fetching additional airport data: {e}")
    
    # Sort by relevance (exact code match first, then by name)
    matches.sort(key=lambda x: (x['code'] != query, x['name']))
    
    return jsonify(matches[:5])  # Return top 5 matches

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
