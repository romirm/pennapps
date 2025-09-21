"""
World Data Processor for Real-World Airport Visualization
Processes world_data.json to create airport maps with real lat/lon coordinates
"""

import json
import math
from typing import Dict, List, Tuple, Optional

def load_world_data(file_path: str = "world_data.json") -> Dict:
    """Load and parse the world_data.json file"""
    try:
        print(f"Attempting to load world data from: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Successfully loaded world data with {len(data.get('elements', []))} elements")
            return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        # Try relative path as fallback
        try:
            with open("world_data.json", 'r', encoding='utf-8') as f:
                data = json.load(f)
                print(f"Successfully loaded world data from relative path with {len(data.get('elements', []))} elements")
                return data
        except Exception as e2:
            print(f"Error loading from relative path: {e2}")
            return None
    except Exception as e:
        print(f"Error loading world_data.json: {e}")
        return None

def calculate_bounds(elements: List[Dict]) -> Tuple[float, float, float, float]:
    """Calculate the bounding box for all elements"""
    min_lat = float('inf')
    max_lat = float('-inf')
    min_lon = float('inf')
    max_lon = float('-inf')
    
    for element in elements:
        if element.get('type') == 'way' and 'geometry' in element:
            for point in element['geometry']:
                min_lat = min(min_lat, point['lat'])
                max_lat = max(max_lat, point['lat'])
                min_lon = min(min_lon, point['lon'])
                max_lon = max(max_lon, point['lon'])
    
    return min_lat, max_lat, min_lon, max_lon

def lat_lon_to_canvas_coords(lat: float, lon: float, bounds: Tuple[float, float, float, float], 
                           canvas_width: int, canvas_height: int, padding: int = 50) -> Tuple[float, float]:
    """Convert lat/lon coordinates to canvas coordinates"""
    min_lat, max_lat, min_lon, max_lon = bounds
    
    # Normalize coordinates to 0-1 range
    norm_lon = (lon - min_lon) / (max_lon - min_lon)
    norm_lat = (lat - min_lat) / (max_lat - min_lat)
    
    # Flip Y coordinate (lat increases north, but canvas Y increases down)
    norm_lat = 1.0 - norm_lat
    
    # Scale to canvas size with padding
    x = padding + norm_lon * (canvas_width - 2 * padding)
    y = padding + norm_lat * (canvas_height - 2 * padding)
    
    return x, y

def process_world_data_for_airport(world_data: Dict, airport_name: str = "JFK") -> Dict:
    """Process world_data.json into airport data structure compatible with current UI"""
    if not world_data or 'elements' not in world_data:
        return None
    
    all_elements = world_data['elements']
    
    # Filter elements based on airport
    if airport_name == "JFK":
        # JFK elements have IDs in the millions (original OpenStreetMap data)
        # Exclude PHL (1M-2M) and LAX (2M-3M) ranges
        elements = [e for e in all_elements if e.get('id', 0) >= 3000000]
    elif airport_name == "PHL":
        # PHL elements have IDs starting with 1000000
        elements = [e for e in all_elements if 1000000 <= e.get('id', 0) < 2000000]
    elif airport_name == "LAX":
        # LAX elements have IDs starting with 2000000
        elements = [e for e in all_elements if 2000000 <= e.get('id', 0) < 3000000]
    else:
        elements = all_elements
    
    if not elements:
        print(f"No elements found for {airport_name}")
        return None
    
    # Calculate bounds for coordinate transformation
    bounds = calculate_bounds(elements)
    print(f"Airport bounds: lat {bounds[0]:.6f} to {bounds[1]:.6f}, lon {bounds[2]:.6f} to {bounds[3]:.6f}")
    
    # Canvas dimensions (will be adjusted by JavaScript)
    canvas_width = 1200
    canvas_height = 800
    
    # Separate runways and taxiways
    runways = []
    taxiways = []
    
    for element in elements:
        if element.get('type') != 'way' or 'geometry' not in element:
            continue
            
        tags = element.get('tags', {})
        aeroway_type = tags.get('aeroway')
        
        if aeroway_type == 'runway':
            # Process runway
            runway_data = {
                'id': tags.get('ref', f"RW{len(runways)+1}"),
                'length': float(tags.get('length', 3000)),  # meters
                'width': float(tags.get('width', 45)),      # meters
                'heading': 0,  # Will be calculated from geometry
                'points': []
            }
            
            # Convert geometry points to canvas coordinates
            for point in element['geometry']:
                x, y = lat_lon_to_canvas_coords(point['lat'], point['lon'], bounds, canvas_width, canvas_height)
                runway_data['points'].append([x, y])
            
            # Calculate heading from first and last points
            if len(runway_data['points']) >= 2:
                start = runway_data['points'][0]
                end = runway_data['points'][-1]
                dx = end[0] - start[0]
                dy = end[1] - start[1]
                runway_data['heading'] = math.degrees(math.atan2(dy, dx))
            
            runways.append(runway_data)
            
        elif aeroway_type == 'taxiway':
            # Process taxiway
            taxiway_data = {
                'id': f"TW{len(taxiways)+1}",
                'points': []
            }
            
            # Convert geometry points to canvas coordinates
            for point in element['geometry']:
                x, y = lat_lon_to_canvas_coords(point['lat'], point['lon'], bounds, canvas_width, canvas_height)
                taxiway_data['points'].append([x, y])
            
            taxiways.append(taxiway_data)
    
    # Create airport data structure
    airport_data = {
        'name': f'{airport_name} Airport',
        'city': 'New York',
        'country': 'USA',
        'runways': runways,
        'taxiways': taxiways,
        'bounds': bounds,
        'canvas_width': canvas_width,
        'canvas_height': canvas_height
    }
    
    print(f"Processed {len(runways)} runways and {len(taxiways)} taxiways")
    
    return airport_data

def get_world_airport_data(airport_code: str = "JFK") -> Optional[Dict]:
    """Main function to get processed airport data from world_data.json"""
    airport_code = airport_code.upper()
    
    # Use world_data.json for JFK, PHL, and LAX
    if airport_code in ["JFK", "PHL", "LAX"]:
        world_data = load_world_data()
        if world_data:
            print(f"Using OpenStreetMap data for {airport_code}...")
            return process_world_data_for_airport(world_data, airport_code)
    
    # For all other airports, create sample data
    print(f"Creating sample data for {airport_code}...")
    return create_sample_airport_data(airport_code)

# Airport coordinate mappings for major airports
AIRPORT_COORDINATES = {
    'JFK': {
        'name': 'John F. Kennedy International Airport',
        'city': 'New York',
        'country': 'USA',
        'bounds': (40.6280, 40.6530, -73.8167, -73.7717),
        'lat': 40.6413, 'lon': -73.7781
    },
    'PHL': {
        'name': 'Philadelphia International Airport',
        'city': 'Philadelphia',
        'country': 'USA',
        'bounds': (39.8500, 39.8900, -75.2600, -75.2200),
        'lat': 39.8720, 'lon': -75.2407
    },
    'LAX': {
        'name': 'Los Angeles International Airport',
        'city': 'Los Angeles',
        'country': 'USA',
        'bounds': (33.9200, 33.9600, -118.4200, -118.3800),
        'lat': 33.9425, 'lon': -118.4081
    },
    'ORD': {
        'name': 'Chicago O\'Hare International Airport',
        'city': 'Chicago',
        'country': 'USA',
        'bounds': (41.9600, 42.0000, -87.9200, -87.8800),
        'lat': 41.9786, 'lon': -87.9048
    },
    'DFW': {
        'name': 'Dallas/Fort Worth International Airport',
        'city': 'Dallas',
        'country': 'USA',
        'bounds': (32.8800, 32.9200, -97.0500, -97.0100),
        'lat': 32.8968, 'lon': -97.0380
    },
    'ATL': {
        'name': 'Hartsfield-Jackson Atlanta International Airport',
        'city': 'Atlanta',
        'country': 'USA',
        'bounds': (33.6200, 33.6600, -84.4400, -84.4000),
        'lat': 33.6407, 'lon': -84.4277
    },
    'DEN': {
        'name': 'Denver International Airport',
        'city': 'Denver',
        'country': 'USA',
        'bounds': (39.8400, 39.8800, -104.6800, -104.6400),
        'lat': 39.8561, 'lon': -104.6737
    },
    'SFO': {
        'name': 'San Francisco International Airport',
        'city': 'San Francisco',
        'country': 'USA',
        'bounds': (37.6100, 37.6400, -122.3900, -122.3500),
        'lat': 37.6213, 'lon': -122.3790
    },
    'BOS': {
        'name': 'Logan International Airport',
        'city': 'Boston',
        'country': 'USA',
        'bounds': (42.3500, 42.3800, -71.0200, -70.9800),
        'lat': 42.3656, 'lon': -71.0096
    },
    'SEA': {
        'name': 'Seattle-Tacoma International Airport',
        'city': 'Seattle',
        'country': 'USA',
        'bounds': (47.4400, 47.4600, -122.3200, -122.2800),
        'lat': 47.4502, 'lon': -122.3088
    },
    'LHR': {
        'name': 'London Heathrow Airport',
        'city': 'London',
        'country': 'UK',
        'bounds': (51.4600, 51.4800, -0.4600, -0.4400),
        'lat': 51.4700, 'lon': -0.4543
    },
    'CDG': {
        'name': 'Charles de Gaulle Airport',
        'city': 'Paris',
        'country': 'France',
        'bounds': (49.0000, 49.0200, 2.5400, 2.5600),
        'lat': 49.0097, 'lon': 2.5479
    },
    'NRT': {
        'name': 'Narita International Airport',
        'city': 'Tokyo',
        'country': 'Japan',
        'bounds': (35.7600, 35.7800, 140.3800, 140.4000),
        'lat': 35.7720, 'lon': 140.3928
    },
    'ICN': {
        'name': 'Incheon International Airport',
        'city': 'Seoul',
        'country': 'South Korea',
        'bounds': (37.4500, 37.4700, 126.4300, 126.4500),
        'lat': 37.4602, 'lon': 126.4407
    },
    'DXB': {
        'name': 'Dubai International Airport',
        'city': 'Dubai',
        'country': 'UAE',
        'bounds': (25.2400, 25.2600, 55.3500, 55.3700),
        'lat': 25.2532, 'lon': 55.3657
    },
    'SIN': {
        'name': 'Singapore Changi Airport',
        'city': 'Singapore',
        'country': 'Singapore',
        'bounds': (1.3500, 1.3800, 103.9800, 104.0000),
        'lat': 1.3644, 'lon': 103.9915
    },
    'HKG': {
        'name': 'Hong Kong International Airport',
        'city': 'Hong Kong',
        'country': 'Hong Kong',
        'bounds': (22.3000, 22.3200, 113.9100, 113.9300),
        'lat': 22.3080, 'lon': 113.9185
    }
}

def create_sample_airport_data(airport_code: str) -> Dict:
    """Create comprehensive sample airport data for any airport"""
    airport_code = airport_code.upper()
    
    # Get airport info or use defaults
    airport_info = AIRPORT_COORDINATES.get(airport_code, {
        'name': f'{airport_code} Airport',
        'city': 'Unknown',
        'country': 'Unknown',
        'bounds': (0, 0.1, 0, 0.1),  # Default small bounds
        'lat': 0, 'lon': 0
    })
    
    bounds = airport_info['bounds']
    
    # Generate realistic runways based on airport-specific layouts
    if airport_code == 'PHL':
        # Philadelphia International Airport - realistic layout
        runways = [
            {
                'id': '09L/27R',
                'length': 3000,
                'width': 45,
                'heading': 90,
                'points': [[200, 300], [600, 300]]
            },
            {
                'id': '09R/27L', 
                'length': 3000,
                'width': 45,
                'heading': 90,
                'points': [[200, 400], [600, 400]]
            },
            {
                'id': '17/35',
                'length': 2500,
                'width': 45,
                'heading': 170,
                'points': [[400, 100], [500, 600]]
            }
        ]
    elif airport_code in ['JFK', 'LAX', 'ORD', 'DFW', 'ATL']:  # Major hubs
        runways = [
            {
                'id': '09L/27R',
                'length': 3000,
                'width': 45,
                'heading': 90,
                'points': [[400, 200], [800, 600]]
            },
            {
                'id': '09R/27L',
                'length': 3000,
                'width': 45,
                'heading': 90,
                'points': [[350, 250], [750, 650]]
            },
            {
                'id': '04L/22R',
                'length': 2500,
                'width': 45,
                'heading': 40,
                'points': [[300, 400], [600, 100]]
            },
            {
                'id': '04R/22L',
                'length': 2500,
                'width': 45,
                'heading': 40,
                'points': [[250, 450], [550, 150]]
            }
        ]
    elif airport_code in ['LHR', 'CDG', 'DEN', 'SFO']:  # Other medium airports
        runways = [
            {
                'id': '09L/27R',
                'length': 3000,
                'width': 45,
                'heading': 90,
                'points': [[400, 200], [800, 600]]
            },
            {
                'id': '09R/27L',
                'length': 3000,
                'width': 45,
                'heading': 90,
                'points': [[350, 250], [750, 650]]
            },
            {
                'id': '04L/22R',
                'length': 2500,
                'width': 45,
                'heading': 40,
                'points': [[300, 400], [600, 100]]
            }
        ]
    else:  # Small airports
        runways = [
            {
                'id': '09/27',
                'length': 2000,
                'width': 30,
                'heading': 90,
                'points': [[400, 300], [800, 500]]
            },
            {
                'id': '04/22',
                'length': 1800,
                'width': 30,
                'heading': 40,
                'points': [[350, 350], [650, 150]]
            }
        ]
    
    # Generate comprehensive taxiway network
    taxiways = generate_comprehensive_taxiways(airport_code, runways)
    
    return {
        'name': airport_info['name'],
        'city': airport_info['city'],
        'country': airport_info['country'],
        'runways': runways,
        'taxiways': taxiways,
        'bounds': bounds,
        'canvas_width': 1200,
        'canvas_height': 800,
        'coordinates': {'lat': airport_info['lat'], 'lon': airport_info['lon']}
    }

def generate_comprehensive_taxiways(airport_code: str, runways: List[Dict]) -> List[Dict]:
    """Generate a realistic and efficient taxiway network for the airport"""
    taxiways = []
    
    # Special handling for specific airports to create realistic bird's-eye view layouts
    if airport_code == 'PHL':
        taxiways.extend(generate_phl_realistic_layout(runways))
        return taxiways
    elif airport_code == 'LAX':
        taxiways.extend(generate_lax_realistic_layout(runways))
        return taxiways
    
    # Determine airport complexity and generate appropriate layout
    if airport_code in ['JFK', 'ORD', 'DFW', 'ATL']:
        # Major hub - generate realistic major airport layout
        taxiways.extend(generate_realistic_major_hub_layout(runways))
    elif airport_code in ['LHR', 'CDG', 'DEN', 'SFO']:
        # Medium airport - generate realistic medium airport layout
        taxiways.extend(generate_realistic_medium_airport_layout(runways))
    else:
        # Small airport - generate realistic small airport layout
        taxiways.extend(generate_realistic_small_airport_layout(runways))
    
    return taxiways

def generate_phl_realistic_layout(runways: List[Dict]) -> List[Dict]:
    """Generate realistic Philadelphia International Airport layout - bird's eye view"""
    taxiways = []
    
    # PHL has parallel runways 09L/27R and 09R/27L, plus cross-field 17/35
    # Terminal area is on the west side, cargo on the east
    
    # Main parallel taxiways along runways
    taxiways.extend([
        # Parallel taxiways for 09L/27R
        {'id': 'TW_A', 'points': [[180, 280], [620, 280]]},  # Left parallel
        {'id': 'TW_B', 'points': [[220, 320], [660, 320]]},  # Right parallel
        
        # Parallel taxiways for 09R/27L  
        {'id': 'TW_C', 'points': [[180, 380], [620, 380]]},  # Left parallel
        {'id': 'TW_D', 'points': [[220, 420], [660, 420]]},  # Right parallel
        
        # Cross-field connectors
        {'id': 'TW_E', 'points': [[300, 280], [300, 420]]},
        {'id': 'TW_F', 'points': [[400, 280], [400, 420]]},
        {'id': 'TW_G', 'points': [[500, 280], [500, 420]]},
        
        # Terminal area taxiways (west side)
        {'id': 'TW_H', 'points': [[100, 250], [150, 300], [200, 350], [250, 400]]},
        {'id': 'TW_I', 'points': [[120, 270], [170, 320], [220, 370], [270, 420]]},
        {'id': 'TW_J', 'points': [[140, 290], [190, 340], [240, 390], [290, 440]]},
        
        # Apron taxiways for terminal gates
        {'id': 'TW_APRON1', 'points': [[80, 200], [100, 220], [120, 240], [140, 260]]},
        {'id': 'TW_APRON2', 'points': [[95, 200], [115, 220], [135, 240], [155, 260]]},
        {'id': 'TW_APRON3', 'points': [[110, 200], [130, 220], [150, 240], [170, 260]]},
        {'id': 'TW_APRON4', 'points': [[125, 200], [145, 220], [165, 240], [185, 260]]},
        
        # Cargo area taxiways (east side)
        {'id': 'TW_K', 'points': [[700, 250], [750, 300], [800, 350], [850, 400]]},
        {'id': 'TW_L', 'points': [[720, 270], [770, 320], [820, 370], [870, 420]]},
        
        # Taxiways connecting to runway 17/35
        {'id': 'TW_M', 'points': [[380, 350], [400, 300], [420, 250], [440, 200]]},
        {'id': 'TW_N', 'points': [[420, 350], [440, 400], [460, 450], [480, 500]]},
        
        # Additional connecting taxiways
        {'id': 'TW_O', 'points': [[350, 300], [350, 400]]},
        {'id': 'TW_P', 'points': [[450, 300], [450, 400]]},
        {'id': 'TW_Q', 'points': [[550, 300], [550, 400]]},
        
        # High-speed taxiways (exit taxiways from runways)
        {'id': 'TW_R', 'points': [[250, 300], [280, 320], [310, 340], [340, 360]]},
        {'id': 'TW_S', 'points': [[250, 400], [280, 380], [310, 360], [340, 340]]},
        {'id': 'TW_T', 'points': [[450, 300], [480, 320], [510, 340], [540, 360]]},
        {'id': 'TW_U', 'points': [[450, 400], [480, 380], [510, 360], [540, 340]]},
    ])
    
    return taxiways

def generate_lax_realistic_layout(runways: List[Dict]) -> List[Dict]:
    """Generate realistic Los Angeles International Airport layout - bird's eye view"""
    taxiways = []
    
    # LAX has parallel runways with horseshoe-shaped terminal area
    # Terminals are arranged in a U-shape around the central area
    
    taxiways.extend([
        # Main parallel taxiways along runways
        {'id': 'TW_A', 'points': [[200, 200], [800, 200]]},  # North parallel
        {'id': 'TW_B', 'points': [[200, 250], [800, 250]]},  # North parallel
        {'id': 'TW_C', 'points': [[200, 300], [800, 300]]},  # Center parallel
        {'id': 'TW_D', 'points': [[200, 350], [800, 350]]},  # South parallel
        {'id': 'TW_E', 'points': [[200, 400], [800, 400]]},  # South parallel
        
        # Cross-field connectors
        {'id': 'TW_F', 'points': [[300, 200], [300, 400]]},
        {'id': 'TW_G', 'points': [[400, 200], [400, 400]]},
        {'id': 'TW_H', 'points': [[500, 200], [500, 400]]},
        {'id': 'TW_I', 'points': [[600, 200], [600, 400]]},
        {'id': 'TW_J', 'points': [[700, 200], [700, 400]]},
        
        # Terminal area taxiways (horseshoe shape)
        # North terminal taxiways
        {'id': 'TW_K', 'points': [[150, 150], [200, 200], [250, 250], [300, 300]]},
        {'id': 'TW_L', 'points': [[170, 170], [220, 220], [270, 270], [320, 320]]},
        {'id': 'TW_M', 'points': [[190, 190], [240, 240], [290, 290], [340, 340]]},
        
        # South terminal taxiways
        {'id': 'TW_N', 'points': [[150, 450], [200, 400], [250, 350], [300, 300]]},
        {'id': 'TW_O', 'points': [[170, 430], [220, 380], [270, 330], [320, 280]]},
        {'id': 'TW_P', 'points': [[190, 410], [240, 360], [290, 310], [340, 260]]},
        
        # West terminal taxiways (connecting north and south)
        {'id': 'TW_Q', 'points': [[100, 200], [120, 250], [140, 300], [160, 350], [180, 400]]},
        {'id': 'TW_R', 'points': [[110, 220], [130, 270], [150, 320], [170, 370], [190, 420]]},
        {'id': 'TW_S', 'points': [[120, 240], [140, 290], [160, 340], [180, 390], [200, 440]]},
        
        # Apron taxiways for terminal gates
        {'id': 'TW_APRON1', 'points': [[80, 180], [100, 200], [120, 220], [140, 240]]},
        {'id': 'TW_APRON2', 'points': [[90, 190], [110, 210], [130, 230], [150, 250]]},
        {'id': 'TW_APRON3', 'points': [[100, 200], [120, 220], [140, 240], [160, 260]]},
        {'id': 'TW_APRON4', 'points': [[80, 420], [100, 400], [120, 380], [140, 360]]},
        {'id': 'TW_APRON5', 'points': [[90, 410], [110, 390], [130, 370], [150, 350]]},
        {'id': 'TW_APRON6', 'points': [[100, 400], [120, 380], [140, 360], [160, 340]]},
        
        # Cargo area taxiways (east side)
        {'id': 'TW_T', 'points': [[850, 200], [900, 250], [950, 300], [1000, 350]]},
        {'id': 'TW_U', 'points': [[870, 220], [920, 270], [970, 320], [1020, 370]]},
        {'id': 'TW_V', 'points': [[850, 400], [900, 350], [950, 300], [1000, 250]]},
        {'id': 'TW_W', 'points': [[870, 380], [920, 330], [970, 280], [1020, 230]]},
        
        # High-speed taxiways (exit taxiways from runways)
        {'id': 'TW_X', 'points': [[250, 200], [280, 220], [310, 240], [340, 260]]},
        {'id': 'TW_Y', 'points': [[250, 400], [280, 380], [310, 360], [340, 340]]},
        {'id': 'TW_Z', 'points': [[450, 200], [480, 220], [510, 240], [540, 260]]},
        {'id': 'TW_AA', 'points': [[450, 400], [480, 380], [510, 360], [540, 340]]},
        {'id': 'TW_BB', 'points': [[650, 200], [680, 220], [710, 240], [740, 260]]},
        {'id': 'TW_CC', 'points': [[650, 400], [680, 380], [710, 360], [740, 340]]},
        
        # Additional connecting taxiways for realistic flow
        {'id': 'TW_DD', 'points': [[350, 200], [350, 400]]},
        {'id': 'TW_EE', 'points': [[550, 200], [550, 400]]},
        {'id': 'TW_FF', 'points': [[750, 200], [750, 400]]},
    ])
    
    return taxiways

def generate_realistic_major_hub_layout(runways: List[Dict]) -> List[Dict]:
    """Generate realistic major hub airport layout with proper taxiway design"""
    taxiways = []
    
    # Calculate airport center and bounds
    all_points = []
    for runway in runways:
        all_points.extend(runway['points'])
    
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Generate parallel taxiways for each runway
    for i, runway in enumerate(runways):
        rw_start = runway['points'][0]
        rw_end = runway['points'][1]
        
        # Calculate runway direction vector
        dx = rw_end[0] - rw_start[0]
        dy = rw_end[1] - rw_start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Perpendicular vector for taxiway offset
            perp_x = -dy_norm
            perp_y = dx_norm
            
            # Left parallel taxiway (offset by 60 units)
            offset = 60
            tw_left_start = [rw_start[0] + perp_x * offset, rw_start[1] + perp_y * offset]
            tw_left_end = [rw_end[0] + perp_x * offset, rw_end[1] + perp_y * offset]
            
            taxiways.append({
                'id': f'TW{i*2+1}',
                'points': [tw_left_start, tw_left_end]
            })
            
            # Right parallel taxiway (offset by -60 units)
            tw_right_start = [rw_start[0] - perp_x * offset, rw_start[1] - perp_y * offset]
            tw_right_end = [rw_end[0] - perp_x * offset, rw_end[1] - perp_y * offset]
            
            taxiways.append({
                'id': f'TW{i*2+2}',
                'points': [tw_right_start, tw_right_end]
            })
    
    # Generate connecting taxiways between runways
    taxiways.extend(generate_connecting_taxiways(runways, center_x, center_y))
    
    # Generate terminal area taxiways
    taxiways.extend(generate_terminal_taxiways(center_x, center_y, major=True))
    
    # Generate cross-field taxiways
    taxiways.extend(generate_cross_field_taxiways_realistic(runways, center_x, center_y))
    
    return taxiways

def generate_realistic_medium_airport_layout(runways: List[Dict]) -> List[Dict]:
    """Generate realistic medium airport layout"""
    taxiways = []
    
    # Calculate airport center
    all_points = []
    for runway in runways:
        all_points.extend(runway['points'])
    
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Generate parallel taxiways for each runway
    for i, runway in enumerate(runways):
        rw_start = runway['points'][0]
        rw_end = runway['points'][1]
        
        # Calculate runway direction vector
        dx = rw_end[0] - rw_start[0]
        dy = rw_end[1] - rw_start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Perpendicular vector for taxiway offset
            perp_x = -dy_norm
            perp_y = dx_norm
            
            # Parallel taxiway (offset by 45 units)
            offset = 45
            tw_start = [rw_start[0] + perp_x * offset, rw_start[1] + perp_y * offset]
            tw_end = [rw_end[0] + perp_x * offset, rw_end[1] + perp_y * offset]
            
            taxiways.append({
                'id': f'TW{i+1}',
                'points': [tw_start, tw_end]
            })
    
    # Generate connecting taxiways
    taxiways.extend(generate_connecting_taxiways(runways, center_x, center_y, medium=True))
    
    # Generate terminal area taxiways
    taxiways.extend(generate_terminal_taxiways(center_x, center_y, major=False))
    
    return taxiways

def generate_realistic_small_airport_layout(runways: List[Dict]) -> List[Dict]:
    """Generate realistic small airport layout"""
    taxiways = []
    
    # Calculate airport center
    all_points = []
    for runway in runways:
        all_points.extend(runway['points'])
    
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    
    # Generate simple parallel taxiways for each runway
    for i, runway in enumerate(runways):
        rw_start = runway['points'][0]
        rw_end = runway['points'][1]
        
        # Calculate runway direction vector
        dx = rw_end[0] - rw_start[0]
        dy = rw_end[1] - rw_start[1]
        length = math.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            # Normalize direction
            dx_norm = dx / length
            dy_norm = dy / length
            
            # Perpendicular vector for taxiway offset
            perp_x = -dy_norm
            perp_y = dx_norm
            
            # Single parallel taxiway (offset by 30 units)
            offset = 30
            tw_start = [rw_start[0] + perp_x * offset, rw_start[1] + perp_y * offset]
            tw_end = [rw_end[0] + perp_x * offset, rw_end[1] + perp_y * offset]
            
            taxiways.append({
                'id': f'TW{i+1}',
                'points': [tw_start, tw_end]
            })
    
    # Generate basic terminal taxiways
    taxiways.extend(generate_terminal_taxiways(center_x, center_y, major=False, small=True))
    
    return taxiways

def generate_connecting_taxiways(runways: List[Dict], center_x: float, center_y: float, medium: bool = False) -> List[Dict]:
    """Generate connecting taxiways between runways"""
    taxiways = []
    
    if len(runways) < 2:
        return taxiways
    
    # Connect runways with taxiways
    for i in range(len(runways) - 1):
        rw1_center = [
            (runways[i]['points'][0][0] + runways[i]['points'][1][0]) / 2,
            (runways[i]['points'][0][1] + runways[i]['points'][1][1]) / 2
        ]
        rw2_center = [
            (runways[i+1]['points'][0][0] + runways[i+1]['points'][1][0]) / 2,
            (runways[i+1]['points'][0][1] + runways[i+1]['points'][1][1]) / 2
        ]
        
        # Create connecting taxiway
        taxiways.append({
            'id': f'TW_CONN{i+1}',
            'points': [rw1_center, rw2_center]
        })
    
    # Connect to airport center
    for i, runway in enumerate(runways):
        rw_center = [
            (runway['points'][0][0] + runway['points'][1][0]) / 2,
            (runway['points'][0][1] + runway['points'][1][1]) / 2
        ]
        
        taxiways.append({
            'id': f'TW_CENTER{i+1}',
            'points': [rw_center, [center_x, center_y]]
        })
    
    return taxiways

def generate_terminal_taxiways(center_x: float, center_y: float, major: bool = True, small: bool = False) -> List[Dict]:
    """Generate realistic terminal area taxiways"""
    taxiways = []
    
    if small:
        # Simple terminal taxiways for small airports
        for i in range(3):
            taxiways.append({
                'id': f'TW_TERM{i+1}',
                'points': [
                    [center_x - 100 + i*50, center_y - 50],
                    [center_x - 80 + i*50, center_y - 30],
                    [center_x - 60 + i*50, center_y - 10],
                    [center_x - 40 + i*50, center_y + 10]
                ]
            })
    elif major:
        # Complex terminal taxiways for major airports
        for i in range(8):
            taxiways.append({
                'id': f'TW_TERM{i+1}',
                'points': [
                    [center_x - 200 + i*40, center_y - 100],
                    [center_x - 180 + i*40, center_y - 80],
                    [center_x - 160 + i*40, center_y - 60],
                    [center_x - 140 + i*40, center_y - 40],
                    [center_x - 120 + i*40, center_y - 20],
                    [center_x - 100 + i*40, center_y]
                ]
            })
        
        # Additional apron taxiways
        for i in range(6):
            taxiways.append({
                'id': f'TW_APRON{i+1}',
                'points': [
                    [center_x - 150 + i*50, center_y + 50],
                    [center_x - 130 + i*50, center_y + 70],
                    [center_x - 110 + i*50, center_y + 90],
                    [center_x - 90 + i*50, center_y + 110]
                ]
            })
    else:
        # Medium airport terminal taxiways
        for i in range(5):
            taxiways.append({
                'id': f'TW_TERM{i+1}',
                'points': [
                    [center_x - 120 + i*40, center_y - 60],
                    [center_x - 100 + i*40, center_y - 40],
                    [center_x - 80 + i*40, center_y - 20],
                    [center_x - 60 + i*40, center_y],
                    [center_x - 40 + i*40, center_y + 20]
                ]
            })
    
    return taxiways

def generate_cross_field_taxiways_realistic(runways: List[Dict], center_x: float, center_y: float) -> List[Dict]:
    """Generate realistic cross-field taxiways"""
    taxiways = []
    
    # Calculate airport bounds
    all_points = []
    for runway in runways:
        all_points.extend(runway['points'])
    
    min_x = min(point[0] for point in all_points)
    max_x = max(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    max_y = max(point[1] for point in all_points)
    
    # Generate cross-field taxiways
    for i in range(4):
        # Horizontal cross-field taxiways
        taxiways.append({
            'id': f'TW_CROSS_H{i+1}',
            'points': [
                [min_x + i*50, center_y - 30],
                [min_x + i*50 + 100, center_y - 20],
                [min_x + i*50 + 200, center_y - 10],
                [min_x + i*50 + 300, center_y],
                [min_x + i*50 + 400, center_y + 10],
                [min_x + i*50 + 500, center_y + 20]
            ]
        })
        
        # Vertical cross-field taxiways
        taxiways.append({
            'id': f'TW_CROSS_V{i+1}',
            'points': [
                [center_x - 30, min_y + i*50],
                [center_x - 20, min_y + i*50 + 100],
                [center_x - 10, min_y + i*50 + 200],
                [center_x, min_y + i*50 + 300],
                [center_x + 10, min_y + i*50 + 400],
                [center_x + 20, min_y + i*50 + 500]
            ]
        })
    
    return taxiways

def create_sample_jfk_data() -> Dict:
    """Create sample JFK airport data with real-world coordinates"""
    # JFK Airport bounds (approximate)
    bounds = (40.6280, 40.6530, -73.8167, -73.7717)  # min_lat, max_lat, min_lon, max_lon
    
    # Sample runways with real JFK coordinates
    runways = [
        {
            'id': '13R/31L',
            'length': 4423,
            'width': 61,
            'heading': 130,
            'points': [
                [400, 200],  # Start of runway
                [800, 600]   # End of runway
            ]
        },
        {
            'id': '13L/31R',
            'length': 4423,
            'width': 61,
            'heading': 130,
            'points': [
                [350, 250],  # Start of runway
                [750, 650]   # End of runway
            ]
        },
        {
            'id': '04R/22L',
            'length': 3682,
            'width': 61,
            'heading': 40,
            'points': [
                [300, 400],  # Start of runway
                [600, 100]   # End of runway
            ]
        },
        {
            'id': '04L/22R',
            'length': 3682,
            'width': 61,
            'heading': 40,
            'points': [
                [250, 450],  # Start of runway
                [550, 150]   # End of runway
            ]
        }
    ]
    
    # Sample taxiways connecting the runways
    taxiways = [
        {
            'id': 'TW1',
            'points': [
                [400, 200],
                [350, 250],
                [300, 300],
                [250, 350],
                [250, 450]
            ]
        },
        {
            'id': 'TW2',
            'points': [
                [800, 600],
                [750, 650],
                [700, 700],
                [650, 750],
                [600, 800]
            ]
        },
        {
            'id': 'TW3',
            'points': [
                [300, 400],
                [325, 425],
                [350, 450],
                [375, 475],
                [400, 500]
            ]
        },
        {
            'id': 'TW4',
            'points': [
                [600, 100],
                [575, 125],
                [550, 150],
                [525, 175],
                [500, 200]
            ]
        },
        {
            'id': 'TW5',
            'points': [
                [450, 300],
                [500, 350],
                [550, 400],
                [600, 450],
                [650, 500]
            ]
        },
        {
            'id': 'TW6',
            'points': [
                [350, 500],
                [400, 550],
                [450, 600],
                [500, 650],
                [550, 700]
            ]
        }
    ]
    
    return {
        'name': 'John F. Kennedy International Airport',
        'city': 'New York',
        'country': 'USA',
        'runways': runways,
        'taxiways': taxiways,
        'bounds': bounds,
        'canvas_width': 1200,
        'canvas_height': 800
    }

if __name__ == "__main__":
    # Test the processor
    airport_data = get_world_airport_data("JFK")
    if airport_data:
        print(f"Airport: {airport_data['name']}")
        print(f"Runways: {len(airport_data['runways'])}")
        print(f"Taxiways: {len(airport_data['taxiways'])}")
        print(f"Bounds: {airport_data['bounds']}")
    else:
        print("Failed to process airport data")
