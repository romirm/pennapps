"""
Flask Integration for data.json Bottleneck Prediction

This shows you how to integrate your data.json file with your Flask app
to get real-time bottleneck predictions.
"""

from flask import Flask, render_template, jsonify, request
import json
import os
from datetime import datetime

# Import the data loader and model
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
def get_bottlenecks_from_data(code):
    """
    API endpoint to get bottleneck predictions from your data.json file
    
    This is where your data.json file gets fed into the model!
    """
    airport_code = code.upper()
    
    try:
        print(f"🔄 Loading data.json for {airport_code}")
        
        # 1. Load data from your data.json file
        raw_data = data_loader.load_data()
        if not raw_data:
            return jsonify({
                'success': False,
                'error': f'Could not load data.json file',
                'airport': airport_code
            })
        
        # 2. Convert to ADS-B format
        adsb_data = data_loader.convert_to_adsb_format(raw_data)
        
        # 3. Create airport configuration
        airport_config = {
            'icao': airport_code,
            'coordinates': get_airport_coordinates(airport_code)
        }
        
        # 4. Feed data into model and get predictions
        print(f"📊 Analyzing {len(adsb_data['aircraft'])} aircraft from data.json...")
        analysis = bottleneck_model.predict_bottlenecks(adsb_data, airport_config)
        
        # 5. Return predictions
        return jsonify({
            'success': True,
            'airport': airport_code,
            'timestamp': datetime.now().isoformat(),
            'aircraft_count': len(adsb_data['aircraft']),
            'data_source': 'data.json',
            'bottlenecks': analysis['bottleneck_predictions'],
            'summary': analysis['airport_summary']
        })
        
    except Exception as e:
        print(f"❌ Error predicting bottlenecks: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'airport': airport_code
        })

@app.route('/api/data/<code>')
def get_data_info(code):
    """
    API endpoint to get information about your data.json file
    
    This shows you what data is available in your file
    """
    airport_code = code.upper()
    
    try:
        # Load data
        raw_data = data_loader.load_data()
        if not raw_data:
            return jsonify({
                'success': False,
                'error': 'Could not load data.json file'
            })
        
        # Convert to ADS-B format
        adsb_data = data_loader.convert_to_adsb_format(raw_data)
        
        # Return data info
        return jsonify({
            'success': True,
            'airport': airport_code,
            'timestamp': datetime.now().isoformat(),
            'aircraft_count': len(adsb_data['aircraft']),
            'data_source': 'data.json',
            'aircraft': adsb_data['aircraft'][:10],  # First 10 aircraft
            'metadata': raw_data.get('metadata', {})
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/api/upload-data', methods=['POST'])
def upload_data():
    """
    API endpoint to upload new data.json file
    
    This allows you to update your data.json file via API
    """
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if file and file.filename.endswith('.json'):
            # Save uploaded file as data.json
            file.save('data.json')
            
            # Verify the file is valid JSON
            with open('data.json', 'r') as f:
                json.load(f)
            
            return jsonify({
                'success': True,
                'message': 'data.json file uploaded successfully',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload a JSON file.'})
            
    except json.JSONDecodeError:
        return jsonify({'success': False, 'error': 'Invalid JSON format'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

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
    print("🚀 Starting Flask app with data.json integration")
    print("📍 Available endpoints:")
    print("   • /airport/<code> - Airport map page")
    print("   • /api/bottlenecks/<code> - Get bottleneck predictions from data.json")
    print("   • /api/data/<code> - Get data.json info")
    print("   • /api/upload-data - Upload new data.json file")
    print("\n🎯 Example usage:")
    print("   • http://localhost:5000/airport/JFK")
    print("   • http://localhost:5000/api/bottlenecks/JFK")
    print("   • http://localhost:5000/api/data/JFK")
    print("\n📁 Put your data.json file in the project root directory")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
