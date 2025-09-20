"""
Airport Data Fetcher using OpenStreetMap
Fetches real runway and taxiway data for any airport using free OSM data
"""

import requests
import json
import re
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies with fallbacks
try:
    import osmnx as ox
    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point, LineString
    OSMNX_AVAILABLE = True
    
    # Configure OSMnx
    try:
        ox.settings.use_cache = True
        ox.settings.log_console = False
    except:
        pass  # OSMnx version compatibility
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    print("Falling back to simplified data fetching...")
    OSMNX_AVAILABLE = False

def get_airport_coordinates(airport_code: str) -> Optional[Tuple[float, float]]:
    """
    Get airport coordinates from OpenFlights database or fallback coordinates
    """
    try:
        # Try OpenFlights database first
        openflights_url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
        response = requests.get(openflights_url, timeout=10)
        
        if response.status_code == 200:
            lines = response.text.split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 7:
                        code = parts[4].strip('"')
                        if code == airport_code:
                            return float(parts[6]), float(parts[7])
    except Exception as e:
        print(f"Error fetching from OpenFlights: {e}")
    
    # Fallback coordinates for major airports
    fallback_coords = {
        'JFK': (40.6413, -73.7781),
        'LAX': (33.9425, -118.4081),
        'PHL': (39.8729, -75.2407),
        'MIA': (25.7959, -80.2870),
        'ORD': (41.9786, -87.9048),
        'DFW': (32.8968, -97.0380),
        'ATL': (33.6407, -84.4277),
        'DEN': (39.8561, -104.6737),
        'SFO': (37.6213, -122.3790),
        'BOS': (42.3656, -71.0096),
        'SEA': (47.4502, -122.3088),
        'LAS': (36.0840, -115.1537),
        'MCO': (28.4312, -81.3081),
        'LHR': (51.4700, -0.4543),
        'CDG': (49.0097, 2.5479),
        'NRT': (35.7720, 140.3928),
        'FRA': (50.0379, 8.5622),
        'AMS': (52.3105, 4.7683),
        'MAD': (40.4983, -3.5676),
        'BCN': (41.2974, 2.0833)
    }
    
    return fallback_coords.get(airport_code.upper())

def fetch_airport_data_simple_api(airport_code: str, radius_meters: int = 5000) -> Dict:
    """
    Fetch airport data using direct Overpass API calls (simpler, more reliable)
    """
    print(f"Fetching OSM data for {airport_code} using Overpass API...")
    
    # Get airport coordinates
    coords = get_airport_coordinates(airport_code)
    if not coords:
        raise ValueError(f"Could not find coordinates for airport {airport_code}")
    
    lat, lon = coords
    print(f"Airport coordinates: {lat}, {lon}")
    
    try:
        # Convert radius to degrees (rough approximation)
        radius_deg = radius_meters / 111000
        
        # Overpass API query for runways and taxiways
        overpass_query = f"""
        [out:json][timeout:25];
        (
          way["aeroway"="runway"]({lat-radius_deg},{lon-radius_deg},{lat+radius_deg},{lon+radius_deg});
          way["aeroway"="taxiway"]({lat-radius_deg},{lon-radius_deg},{lat+radius_deg},{lon+radius_deg});
        );
        out geom;
        """
        
        overpass_url = "http://overpass-api.de/api/interpreter"
        
        response = requests.post(overpass_url, data=overpass_query, timeout=30)
        
        if response.status_code != 200:
            print(f"Overpass API error: {response.status_code}")
            return None
            
        data = response.json()
        
        runways = []
        taxiways = []
        
        for element in data.get('elements', []):
            if element['type'] == 'way':
                tags = element.get('tags', {})
                geometry = element.get('geometry', [])
                
                if len(geometry) < 2:
                    continue
                
                # Convert coordinates to relative points
                relative_points = []
                for coord in geometry:
                    rel_x = (coord['lon'] - lon) * 111000  # Convert to meters
                    rel_y = (coord['lat'] - lat) * 111000
                    relative_points.append((rel_x, rel_y))
                
                if tags.get('aeroway') == 'runway':
                    # Calculate runway properties
                    start_point = geometry[0]
                    end_point = geometry[-1]
                    
                    # Calculate length and heading
                    import math
                    length = math.sqrt(
                        (end_point['lon'] - start_point['lon'])**2 + 
                        (end_point['lat'] - start_point['lat'])**2
                    ) * 111000  # Convert to meters
                    
                    heading = math.degrees(math.atan2(
                        end_point['lon'] - start_point['lon'],
                        end_point['lat'] - start_point['lat']
                    ))
                    if heading < 0:
                        heading += 360
                    
                    runway_id = tags.get('ref', f'RWY_{len(runways)+1}')
                    width = tags.get('width', '45')
                    
                    # Parse width
                    if isinstance(width, str):
                        numbers = re.findall(r'\d+', width)
                        if numbers:
                            width = int(numbers[0])
                            if 'ft' in str(width).lower():
                                width = int(width * 0.3048)  # Convert feet to meters
                        else:
                            width = 45
                    else:
                        width = int(width) if width else 45
                    
                    runways.append({
                        'id': str(runway_id),
                        'length': int(length),
                        'width': width,
                        'heading': int(heading),
                        'coordinates': geometry
                    })
                
                elif tags.get('aeroway') == 'taxiway':
                    taxiway_id = tags.get('ref', f'TWY_{len(taxiways)+1}')
                    
                    taxiways.append({
                        'id': str(taxiway_id),
                        'points': relative_points,
                        'coordinates': geometry
                    })
        
        print(f"Found {len(runways)} runways and {len(taxiways)} taxiways")
        
        return {
            'airport_code': airport_code,
            'coordinates': {'lat': lat, 'lng': lon},
            'runways': runways,
            'taxiways': taxiways,
            'metadata': {
                'source': 'OpenStreetMap Overpass API',
                'radius_meters': radius_meters,
                'total_runways': len(runways),
                'total_taxiways': len(taxiways)
            }
        }
        
    except Exception as e:
        print(f"Error fetching OSM data via API: {e}")
        return None

def fetch_airport_data_osm(airport_code: str, radius_meters: int = 5000) -> Dict:
    """
    Fetch airport runway and taxiway data from OpenStreetMap
    Uses simplified Overpass API approach for better reliability
    
    Args:
        airport_code: ICAO or IATA airport code (e.g., 'JFK', 'LAX')
        radius_meters: Search radius in meters around airport center
    
    Returns:
        Dictionary containing runways, taxiways, and metadata
    """
    print(f"Fetching OSM data for {airport_code}...")
    
    # Try the simple API approach first
    try:
        return fetch_airport_data_simple_api(airport_code, radius_meters)
    except Exception as e:
        print(f"Simple API approach failed: {e}")
    
    # Fallback to OSMnx if available
    if not OSMNX_AVAILABLE:
        print("OSMnx not available, using fallback data")
        return None
    
    try:
        # Get airport coordinates
        coords = get_airport_coordinates(airport_code)
        if not coords:
            raise ValueError(f"Could not find coordinates for airport {airport_code}")
        
        lat, lon = coords
        print(f"Airport coordinates: {lat}, {lon}")
        
        # Create a point for the airport center
        airport_point = Point(lon, lat)
        
        # Define search area (bounding box)
        radius_deg = radius_meters / 111000  # Rough conversion: 1 degree ≈ 111km
        bbox = (lon - radius_deg, lat - radius_deg, lon + radius_deg, lat + radius_deg)
        
        # Fetch runways from OSM
        print("Fetching runways with OSMnx...")
        try:
            runways_gdf = ox.features_from_point(
                (lat, lon), 
                dist=radius_meters,
                tags={'aeroway': 'runway'}
            )
        except:
            # Fallback method
            runways_gdf = ox.features_from_bbox(
                lat - radius_deg, lon - radius_deg, 
                lat + radius_deg, lon + radius_deg,
                tags={'aeroway': 'runway'}
            )
        
        # Fetch taxiways from OSM
        print("Fetching taxiways with OSMnx...")
        try:
            taxiways_gdf = ox.features_from_point(
                (lat, lon), 
                dist=radius_meters,
                tags={'aeroway': 'taxiway'}
            )
        except:
            # Fallback method
            taxiways_gdf = ox.features_from_bbox(
                lat - radius_deg, lon - radius_deg, 
                lat + radius_deg, lon + radius_deg,
                tags={'aeroway': 'taxiway'}
            )
        
        # Process runways
        runways = []
        if not runways_gdf.empty:
            for idx, runway in runways_gdf.iterrows():
                if hasattr(runway.geometry, 'coords'):
                    coords_list = list(runway.geometry.coords)
                    if len(coords_list) >= 2:
                        # Calculate runway properties
                        start_point = coords_list[0]
                        end_point = coords_list[-1]
                        
                        # Calculate length and heading
                        length = runway.geometry.length * 111000  # Convert to meters
                        heading = np.degrees(np.arctan2(
                            end_point[0] - start_point[0],
                            end_point[1] - start_point[1]
                        ))
                        if heading < 0:
                            heading += 360
                        
                        # Get runway ID from OSM tags
                        runway_id = runway.get('ref', f'RWY_{len(runways)+1}')
                        if pd.isna(runway_id) or runway_id is None:
                            runway_id = f'RWY_{len(runways)+1}'
                        if '/' in str(runway_id):
                            runway_id = str(runway_id).replace('/', '/')
                        
                        # Handle width parsing (could be string like "150 ft" or number)
                        width = runway.get('width', 45)
                        if pd.isna(width) or width is None:
                            width = 45
                        elif isinstance(width, str):
                            # Try to extract number from string like "150 ft"
                            import re
                            numbers = re.findall(r'\d+', str(width))
                            if numbers:
                                width = int(numbers[0])
                                if 'ft' in str(width).lower():
                                    width = int(width * 0.3048)  # Convert feet to meters
                            else:
                                width = 45
                        else:
                            width = int(width)
                        
                        runways.append({
                            'id': str(runway_id),
                            'length': int(length),
                            'width': width,
                            'heading': int(heading),
                            'coordinates': coords_list
                        })
        
        # Process taxiways
        taxiways = []
        if not taxiways_gdf.empty:
            for idx, taxiway in taxiways_gdf.iterrows():
                if hasattr(taxiway.geometry, 'coords'):
                    coords_list = list(taxiway.geometry.coords)
                    if len(coords_list) >= 2:
                        # Get taxiway ID from OSM tags
                        taxiway_id = taxiway.get('ref', f'TWY_{len(taxiways)+1}')
                        if pd.isna(taxiway_id) or taxiway_id is None:
                            taxiway_id = f'TWY_{len(taxiways)+1}'
                        
                        # Convert coordinates to relative points for visualization
                        # Normalize coordinates relative to airport center
                        relative_points = []
                        for coord in coords_list:
                            rel_x = (coord[0] - lon) * 111000  # Convert to meters
                            rel_y = (coord[1] - lat) * 111000
                            relative_points.append((rel_x, rel_y))
                        
                        taxiways.append({
                            'id': str(taxiway_id),
                            'points': relative_points,
                            'coordinates': coords_list
                        })
        
        print(f"Found {len(runways)} runways and {len(taxiways)} taxiways")
        
        return {
            'airport_code': airport_code,
            'coordinates': {'lat': lat, 'lng': lon},
            'runways': runways,
            'taxiways': taxiways,
            'metadata': {
                'source': 'OpenStreetMap OSMnx',
                'radius_meters': radius_meters,
                'total_runways': len(runways),
                'total_taxiways': len(taxiways)
            }
        }
        
    except Exception as e:
        print(f"Error fetching OSM data with OSMnx: {e}")
        return None

def plot_airport_map(airport_data: Dict, save_path: Optional[str] = None) -> None:
    """
    Plot airport runways and taxiways using matplotlib
    """
    if not airport_data:
        print("No airport data to plot")
        return
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot taxiways in blue
    for taxiway in airport_data['taxiways']:
        if len(taxiway['points']) >= 2:
            x_coords = [point[0] for point in taxiway['points']]
            y_coords = [point[1] for point in taxiway['points']]
            ax.plot(x_coords, y_coords, 'b-', linewidth=1, alpha=0.7)
    
    # Plot runways in red
    for runway in airport_data['runways']:
        if len(runway['coordinates']) >= 2:
            x_coords = [coord[0] for coord in runway['coordinates']]
            y_coords = [coord[1] for coord in runway['coordinates']]
            ax.plot(x_coords, y_coords, 'r-', linewidth=3, alpha=0.8)
            
            # Add runway ID label
            mid_x = sum(x_coords) / len(x_coords)
            mid_y = sum(y_coords) / len(y_coords)
            ax.text(mid_x, mid_y, runway['id'], 
                   ha='center', va='center', 
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                   fontsize=8, fontweight='bold')
    
    ax.set_title(f"Airport Map: {airport_data['airport_code']}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitude (degrees)")
    ax.grid(True, alpha=0.3)
    ax.legend(['Taxiways', 'Runways'], loc='upper right')
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Map saved to {save_path}")
    
    plt.show()

def convert_to_app_format(airport_data: Dict) -> Dict:
    """
    Convert OSM airport data to the format expected by the Flask app
    """
    if not airport_data:
        return None
    
    # Convert runways to app format
    app_runways = []
    for runway in airport_data['runways']:
        app_runways.append({
            'id': runway['id'],
            'length': runway['length'],
            'width': runway['width'],
            'heading': runway['heading']
        })
    
    # Convert taxiways to app format (simplified coordinates)
    app_taxiways = []
    for taxiway in airport_data['taxiways']:
        # Convert meter coordinates to simplified grid coordinates
        simplified_points = []
        for point in taxiway['points']:
            # Scale down coordinates for visualization
            x = int(point[0] / 10)  # Scale factor
            y = int(point[1] / 10)
            simplified_points.append((x, y))
        
        app_taxiways.append({
            'id': taxiway['id'],
            'points': simplified_points
        })
    
    return {
        'name': f"{airport_data['airport_code']} Airport",
        'city': 'Unknown',
        'country': 'Unknown',
        'coordinates': airport_data['coordinates'],
        'runways': app_runways,
        'taxiways': app_taxiways,
        'metadata': airport_data['metadata']
    }

def get_airport_data_for_app(airport_code: str) -> Dict:
    """
    Main function to get airport data in app format
    """
    try:
        print(f"🔄 Attempting to fetch OSM data for {airport_code}...")
        
        # Fetch OSM data
        osm_data = fetch_airport_data_osm(airport_code)
        
        if osm_data and osm_data.get('runways') and osm_data.get('taxiways'):
            print(f"✅ Successfully fetched OSM data: {len(osm_data['runways'])} runways, {len(osm_data['taxiways'])} taxiways")
            # Convert to app format
            app_data = convert_to_app_format(osm_data)
            return app_data
        else:
            print(f"❌ No OSM data found for {airport_code}")
            return None
            
    except Exception as e:
        print(f"❌ Error getting airport data for {airport_code}: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Test with JFK
    airport_code = "JFK"
    
    print(f"Testing airport data fetcher with {airport_code}")
    
    # Get data
    airport_data = get_airport_data_for_app(airport_code)
    
    if airport_data:
        print(f"\nAirport: {airport_data['name']}")
        print(f"Runways: {len(airport_data['runways'])}")
        print(f"Taxiways: {len(airport_data['taxiways'])}")
        
        # Print runway details
        print("\nRunways:")
        for runway in airport_data['runways']:
            print(f"  {runway['id']}: {runway['length']}m x {runway['width']}m, heading {runway['heading']}°")
        
        # Print taxiway details
        print(f"\nTaxiways (showing first 5):")
        for taxiway in airport_data['taxiways'][:5]:
            print(f"  {taxiway['id']}: {len(taxiway['points'])} points")
    else:
        print("Failed to fetch airport data")
