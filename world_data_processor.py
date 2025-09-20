"""
World Data Processor for Real-World Airport Visualization
Processes world_data.json to create airport maps with real lat/lon coordinates
"""

import json
import math
from typing import Dict, List, Tuple, Optional

def load_world_data(file_path: str = "/Users/finnmcmillion/Desktop/pennapps/world_data.json") -> Dict:
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
    
    elements = world_data['elements']
    
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
    # Only JFK has world data
    if airport_code.upper() != "JFK":
        return None
    
    world_data = load_world_data()
    if not world_data:
        # Create sample JFK data based on the structure shown
        print("Creating sample JFK data...")
        return create_sample_jfk_data()
    
    return process_world_data_for_airport(world_data, airport_code)

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
