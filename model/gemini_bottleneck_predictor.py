"""
Gemini-Powered Bottleneck Predictor
A modern, AI-driven airport bottleneck prediction system using Google's Gemini API
"""

import json
import time
import math
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import asyncio
import aiohttp
import hashlib
from functools import lru_cache

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not available. Install with: pip install google-generativeai")


@dataclass
class AircraftData:
    """Structured aircraft data for analysis"""
    flight: str
    lat: float
    lon: float
    altitude: Any  # Can be int, float, or 'ground'
    speed: float
    heading: float
    timestamp: Optional[str] = None


@dataclass
class BottleneckPrediction:
    """Structured bottleneck prediction result matching inference.py format"""
    bottleneck_id: str
    coordinates: List[float]  # [lat, lng]
    timestamp: str
    type: str  # "runway", "taxiway", "approach", "departure"
    severity: int  # 1-5
    duration: float  # minutes
    confidence: float  # 0.0-1.0
    aircraft_count: int
    aircraft_affected: List[Dict]  # List of aircraft details


@dataclass
class TrafficAnalysis:
    """Traffic density and pattern analysis"""
    total_aircraft: int
    ground_aircraft: int
    airborne_aircraft: int
    density_score: float
    hotspots: List[Dict]
    congestion_areas: List[Dict]
    movement_patterns: Dict


class GeminiBottleneckPredictor:
    """Main predictor class using Google's Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the predictor with Gemini API"""
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        # Configure Gemini only if API key is available
        if self.api_key and GEMINI_AVAILABLE:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Gemini Bottleneck Predictor initialized successfully")
        else:
            self.model = None
            if not self.api_key:
                print("⚠️  GEMINI_API_KEY not set - using fallback mode")
            if not GEMINI_AVAILABLE:
                print("⚠️  google-generativeai not available - using fallback mode")
        
        self.results_file = "results.txt"
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles"""
        R = 3440.065  # Earth's radius in nautical miles
        
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def analyze_traffic_density(self, aircraft_data: List[Dict]) -> TrafficAnalysis:
        """Analyze traffic density patterns and hotspots"""
        if not aircraft_data:
            return TrafficAnalysis(
                total_aircraft=0,
                ground_aircraft=0,
                airborne_aircraft=0,
                density_score=0,
                hotspots=[],
                congestion_areas=[],
                movement_patterns={}
            )
        
        # Categorize aircraft
        ground_aircraft = [ac for ac in aircraft_data if ac.get('altitude') == 'ground' or ac.get('altitude') == 0]
        airborne_aircraft = [ac for ac in aircraft_data if isinstance(ac.get('altitude'), (int, float)) and ac.get('altitude', 0) > 0]
        
        # Find hotspots using distance-based clustering
        hotspots = self._find_hotspots(aircraft_data)
        
        # Analyze movement patterns
        movement_patterns = self._analyze_movement_patterns(aircraft_data)
        
        # Calculate density score (ultra granular for small bottlenecks)
        total_aircraft = len(aircraft_data)
        ground_ratio = len(ground_aircraft) / max(total_aircraft, 1)
        hotspot_factor = len(hotspots) * 3  # Very low
        aircraft_factor = total_aircraft * 1.5  # Very low aircraft factor
        
        density_score = min(100, (ground_ratio * 15) + hotspot_factor + aircraft_factor)  # Ultra granular
        
        # Identify congestion areas (more sensitive)
        congestion_areas = [h for h in hotspots if h['aircraft_count'] >= 2]
        
        return TrafficAnalysis(
            total_aircraft=total_aircraft,
            ground_aircraft=len(ground_aircraft),
            airborne_aircraft=len(airborne_aircraft),
            density_score=density_score,
            hotspots=hotspots[:5],  # Top 5 hotspots
            congestion_areas=congestion_areas,
            movement_patterns=movement_patterns
        )
    
    def _find_hotspots(self, aircraft_data: List[Dict]) -> List[Dict]:
        """Find geographic hotspots where aircraft are clustered"""
        hotspots = []
        
        if len(aircraft_data) < 2:
            return hotspots
        
        for i, ac1 in enumerate(aircraft_data):
            nearby_aircraft = []
            
            for j, ac2 in enumerate(aircraft_data):
                if i != j:
                    dist = self.calculate_distance(
                        ac1.get('lat', 0), ac1.get('lon', 0),
                        ac2.get('lat', 0), ac2.get('lon', 0)
                    )
                    if dist < 5:  # Within 5 nautical miles (more sensitive)
                        nearby_aircraft.append(ac2)
            
            if len(nearby_aircraft) >= 1:  # At least 2 aircraft total (more sensitive)
                hotspot = {
                    'center_lat': ac1.get('lat'),
                    'center_lon': ac1.get('lon'),
                    'aircraft_count': len(nearby_aircraft) + 1,
                    'aircraft': [ac1.get('flight', 'Unknown')] + [ac.get('flight', 'Unknown') for ac in nearby_aircraft],
                    'radius_nm': 3,
                    'severity': min(100, len(nearby_aircraft) * 20)
                }
                hotspots.append(hotspot)
        
        # Remove duplicates and sort by severity
        unique_hotspots = []
        seen_centers = set()
        
        for hotspot in hotspots:
            center_key = (round(hotspot['center_lat'], 3), round(hotspot['center_lon'], 3))
            if center_key not in seen_centers:
                unique_hotspots.append(hotspot)
                seen_centers.add(center_key)
        
        return sorted(unique_hotspots, key=lambda x: x['severity'], reverse=True)
    
    def _analyze_movement_patterns(self, aircraft_data: List[Dict]) -> Dict:
        """Analyze aircraft movement patterns for bottleneck indicators"""
        patterns = {
            'stationary': 0,
            'slow_moving': 0,
            'normal_moving': 0,
            'fast_moving': 0
        }
        
        for ac in aircraft_data:
            speed = ac.get('speed', 0)
            
            if speed == 0:
                patterns['stationary'] += 1
            elif 0 < speed <= 20:
                patterns['slow_moving'] += 1
            elif 20 < speed <= 100:
                patterns['normal_moving'] += 1
            else:
                patterns['fast_moving'] += 1
        
        return patterns
    
    def create_analysis_prompt(self, aircraft_data: List[Dict], traffic_analysis: TrafficAnalysis, airport_code: str) -> str:
        """Create a structured prompt for Gemini analysis"""
        
        # Prepare aircraft summary
        aircraft_summary = []
        for i, ac in enumerate(aircraft_data[:15]):  # Limit to first 15 for prompt size
            aircraft_summary.append({
                'flight': ac.get('flight', 'Unknown'),
                'position': f"{ac.get('lat', 0):.4f}, {ac.get('lon', 0):.4f}",
                'altitude': ac.get('altitude', 0),
                'speed': ac.get('speed', 0),
                'heading': ac.get('heading', 0)
            })
        
        prompt = f"""
You are an expert aviation traffic analyst. Analyze the following airport traffic data for {airport_code} and predict potential bottlenecks.

TRAFFIC SUMMARY:
- Total Aircraft: {traffic_analysis.total_aircraft}
- Ground Aircraft: {traffic_analysis.ground_aircraft}
- Airborne Aircraft: {traffic_analysis.airborne_aircraft}
- Density Score: {traffic_analysis.density_score:.1f}/100
- Hotspots Detected: {len(traffic_analysis.hotspots)}
- Congestion Areas: {len(traffic_analysis.congestion_areas)}

MOVEMENT PATTERNS:
- Stationary: {traffic_analysis.movement_patterns.get('stationary', 0)}
- Slow Moving: {traffic_analysis.movement_patterns.get('slow_moving', 0)}
- Normal Moving: {traffic_analysis.movement_patterns.get('normal_moving', 0)}
- Fast Moving: {traffic_analysis.movement_patterns.get('fast_moving', 0)}

HOTSPOTS:
{json.dumps(traffic_analysis.hotspots, indent=2)}

AIRCRAFT DETAILS:
{json.dumps(aircraft_summary, indent=2)}

ANALYSIS REQUIREMENTS (SENSITIVE DETECTION):
1. Calculate bottleneck severity score (0-100) based on:
   - Ground aircraft density (detect even small clusters)
   - Geographic clustering (5nm radius, 2+ aircraft)
   - Movement patterns (including slow-moving aircraft)
   - Historical airport capacity

2. Identify specific bottleneck locations:
   - Runway areas (even minor delays)
   - Taxiway intersections (small clusters)
   - Terminal gates (gate congestion)
   - Approach corridors (approach delays)

3. Estimate bottleneck duration based on:
   - Current traffic volume (detect 2-3 minute delays)
   - Typical resolution times (shorter durations)
   - Airport operational capacity

4. Generate actionable recommendations:
   - Immediate actions (0-5 minutes)
   - Short-term actions (5-20 minutes)
   - Preventive measures

5. Assess confidence level based on:
   - Data completeness
   - Traffic pattern clarity
   - Historical accuracy

REQUIRED OUTPUT FORMAT (JSON):
{{
    "severity_score": 75,
    "risk_level": "High",
    "bottleneck_locations": ["Runway 4L", "Terminal A Gates", "Taxiway Charlie"],
    "estimated_duration": "45-60 minutes",
    "affected_aircraft": ["DAL123", "UAL456", "AAL789"],
    "recommendations": [
        "Implement ground stop for departing aircraft",
        "Open additional taxiway routes",
        "Increase ground crew staffing",
        "Consider runway configuration change"
    ],
    "confidence": 85,
    "analysis_notes": "High ground traffic density with multiple congestion hotspots detected"
}}

Provide your analysis in the exact JSON format specified above.
"""
        
        return prompt
    
    async def predict_bottlenecks(self, aircraft_data: List[Dict], airport_code: str = "JFK") -> BottleneckPrediction:
        """Main prediction method using Gemini AI"""
        
        if not aircraft_data:
            return BottleneckPrediction(
                bottleneck_id=f"empty_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
                coordinates=[0.0, 0.0],
                timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                type="unknown",
                severity=1,
                duration=0.0,
                confidence=0.0,
                aircraft_count=0,
                aircraft_affected=[]
            )
        
        try:
            # Analyze traffic patterns
            traffic_analysis = self.analyze_traffic_density(aircraft_data)
            
            # Use fallback if no Gemini model available
            if not self.model:
                return self._fallback_prediction(traffic_analysis, airport_code, aircraft_data)
            
            # Create analysis prompt
            prompt = self.create_analysis_prompt(aircraft_data, traffic_analysis, airport_code)
            
            # Get Gemini analysis
            print("🤖 Running Gemini AI analysis...")
            response = await self._get_gemini_response(prompt)
            
            # Parse response
            try:
                analysis_result = json.loads(response)
                
                # Create aircraft affected list in the required format
                aircraft_affected = []
                for i, ac in enumerate(aircraft_data[:10]):  # Limit to first 10 aircraft
                    aircraft_affected.append({
                        "flight_id": ac.get('flight', f'UNKNOWN_{i}'),
                        "aircraft_type": self.estimate_aircraft_type(ac.get('flight', '')),
                        "position": [ac.get('lat', 0.0), ac.get('lon', 0.0)],
                        "time": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                    })
                
                # Calculate bottleneck center coordinates
                if aircraft_data:
                    avg_lat = sum(ac.get('lat', 0) for ac in aircraft_data) / len(aircraft_data)
                    avg_lng = sum(ac.get('lon', 0) for ac in aircraft_data) / len(aircraft_data)
                else:
                    avg_lat, avg_lng = 0.0, 0.0
                
                # Convert severity score to 1-5 scale
                severity_score = analysis_result.get('severity_score', 0)
                severity = min(5, max(1, int(severity_score / 20)))  # Convert 0-100 to 1-5
                
                # Extract duration from estimated_duration string
                duration_str = analysis_result.get('estimated_duration', '0 minutes')
                duration = self._parse_duration(duration_str)
                
                return BottleneckPrediction(
                    bottleneck_id=f"{analysis_result.get('risk_level', 'unknown').lower()}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
                    coordinates=[avg_lat, avg_lng],
                    timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    type=analysis_result.get('bottleneck_locations', ['unknown'])[0].lower() if analysis_result.get('bottleneck_locations') else 'unknown',
                    severity=severity,
                    duration=duration,
                    confidence=analysis_result.get('confidence', 0) / 100.0,  # Convert to 0.0-1.0
                    aircraft_count=len(aircraft_data),
                    aircraft_affected=aircraft_affected
                )
                
            except json.JSONDecodeError as e:
                print(f"⚠️ Failed to parse Gemini response as JSON: {e}")
                return self._fallback_prediction(traffic_analysis, airport_code, aircraft_data)
                
        except Exception as e:
            print(f"⚠️ Gemini analysis failed: {e}")
            return self._fallback_prediction(traffic_analysis, airport_code, aircraft_data)
    
    async def _get_gemini_response(self, prompt: str) -> str:
        """Get response from Gemini API with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"⚠️ Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
    
    def _parse_duration(self, duration_str: str) -> float:
        """Parse duration string to float minutes"""
        try:
            if 'minutes' in duration_str.lower():
                return float(duration_str.split()[0])
            elif 'hours' in duration_str.lower():
                return float(duration_str.split()[0]) * 60
            else:
                return 0.0
        except:
            return 0.0
    
    def estimate_aircraft_type(self, callsign: str) -> str:
        """Estimate aircraft type from callsign"""
        callsign_clean = callsign.strip().upper()
        
        if callsign_clean.startswith(('DAL', 'DL')):
            return 'B737'
        elif callsign_clean.startswith(('UAL', 'UA')):
            return 'B737'
        elif callsign_clean.startswith(('AAL', 'AA')):
            return 'B737'
        elif callsign_clean.startswith(('JBU', 'B6')):
            return 'A320'
        elif callsign_clean.startswith(('SWA', 'WN')):
            return 'B737'
        else:
            return 'B737'  # Default
    
    def _fallback_prediction(self, traffic_analysis: TrafficAnalysis, airport_code: str, aircraft_data: List[Dict] = None) -> BottleneckPrediction:
        """Fallback prediction when Gemini is unavailable"""
        
        # Simple heuristic-based prediction
        severity_score = min(100, traffic_analysis.density_score)
        
        if severity_score >= 60:
            severity = 5
            duration = 45.0
            bottleneck_type = "runway"
        elif severity_score >= 45:
            severity = 4
            duration = 20.0
            bottleneck_type = "taxiway"
        elif severity_score >= 30:
            severity = 3
            duration = 10.0
            bottleneck_type = "approach"
        elif severity_score >= 18:
            severity = 2
            duration = 3.0
            bottleneck_type = "departure"
        else:
            severity = 1
            duration = 2.0
            bottleneck_type = "gate"
        
        # Use real aircraft data if available
        aircraft_affected = []
        if aircraft_data:
            for i, ac in enumerate(aircraft_data[:10]):  # Limit to first 10
                aircraft_affected.append({
                    "flight_id": ac.get('flight', f'UNKNOWN_{i+1}'),
                    "aircraft_type": self.estimate_aircraft_type(ac.get('flight', '')),
                    "position": [ac.get('lat', 0.0), ac.get('lon', 0.0)],
                    "time": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                })
        else:
            # Create sample aircraft affected list
            for i in range(min(5, traffic_analysis.total_aircraft)):
                aircraft_affected.append({
                    "flight_id": f"SAMPLE_{i+1}",
                    "aircraft_type": "B737",
                    "position": [40.6413 + (i * 0.001), -73.7781 + (i * 0.001)],
                    "time": datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
                })
        
        # Calculate center coordinates
        if aircraft_data:
            avg_lat = sum(ac.get('lat', 0) for ac in aircraft_data) / len(aircraft_data)
            avg_lng = sum(ac.get('lon', 0) for ac in aircraft_data) / len(aircraft_data)
        else:
            avg_lat, avg_lng = 40.6413, -73.7781  # Default JFK coordinates
        
        return BottleneckPrediction(
            bottleneck_id=f"{bottleneck_type}_{datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}",
            coordinates=[avg_lat, avg_lng],
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            type=bottleneck_type,
            severity=severity,
            duration=duration,
            confidence=0.6,  # Lower confidence for fallback
            aircraft_count=traffic_analysis.total_aircraft,
            aircraft_affected=aircraft_affected
        )
    
    def save_results(self, prediction: BottleneckPrediction, traffic_analysis: TrafficAnalysis, airport_code: str):
        """Save prediction results to results.txt"""
        try:
            with open(self.results_file, 'a') as f:
                f.write(f"\n=== Gemini Bottleneck Prediction - {prediction.timestamp} ===\n")
                f.write(f"Bottleneck ID: {prediction.bottleneck_id}\n")
                f.write(f"Coordinates: {prediction.coordinates[0]:.6f}, {prediction.coordinates[1]:.6f}\n")
                f.write(f"Type: {prediction.type}\n")
                f.write(f"Severity: {prediction.severity}/5\n")
                f.write(f"Duration: {prediction.duration:.1f} minutes\n")
                f.write(f"Confidence: {prediction.confidence:.2f}\n")
                f.write(f"Aircraft Count: {prediction.aircraft_count}\n")
                
                f.write(f"\nTraffic Analysis:\n")
                f.write(f"  - Total Aircraft: {traffic_analysis.total_aircraft}\n")
                f.write(f"  - Ground Aircraft: {traffic_analysis.ground_aircraft}\n")
                f.write(f"  - Airborne Aircraft: {traffic_analysis.airborne_aircraft}\n")
                f.write(f"  - Density Score: {traffic_analysis.density_score:.1f}/100\n")
                f.write(f"  - Hotspots: {len(traffic_analysis.hotspots)}\n")
                f.write(f"  - Congestion Areas: {len(traffic_analysis.congestion_areas)}\n")
                
                if prediction.aircraft_affected:
                    f.write(f"\nAffected Aircraft:\n")
                    for aircraft in prediction.aircraft_affected:
                        f.write(f"  - {aircraft['flight_id']} ({aircraft['aircraft_type']})\n")
                        f.write(f"    Position: {aircraft['position'][0]:.6f}, {aircraft['position'][1]:.6f}\n")
                        f.write(f"    Time: {aircraft['time']}\n")
                
                f.write("\n" + "="*60 + "\n")
                
        except Exception as e:
            print(f"Error saving results: {e}")
    
    async def predict_and_save(self, aircraft_data: List[Dict], airport_code: str = "JFK") -> BottleneckPrediction:
        """Main method to predict bottlenecks and save results"""
        
        # Get prediction
        prediction = await self.predict_bottlenecks(aircraft_data, airport_code)
        
        # Analyze traffic for context
        traffic_analysis = self.analyze_traffic_density(aircraft_data)
        
        # Save results
        self.save_results(prediction, traffic_analysis, airport_code)
        
        return prediction


# Example usage and testing
async def main():
    """Example usage of the Gemini Bottleneck Predictor"""
    
    # Sample aircraft data
    sample_aircraft = [
        {
            "flight": "DAL123",
            "lat": 40.6413,
            "lon": -73.7781,
            "altitude": "ground",
            "speed": 0,
            "heading": 90
        },
        {
            "flight": "UAL456",
            "lat": 40.6420,
            "lon": -73.7790,
            "altitude": 2500,
            "speed": 180,
            "heading": 270
        },
        {
            "flight": "AAL789",
            "lat": 40.6415,
            "lon": -73.7785,
            "altitude": "ground",
            "speed": 5,
            "heading": 180
        }
    ]
    
    try:
        # Initialize predictor
        predictor = GeminiBottleneckPredictor()
        
        # Run prediction
        result = await predictor.predict_and_save(sample_aircraft, "JFK")
        
        print("✅ Prediction completed successfully!")
        print(f"Severity Score: {result.severity_score}")
        print(f"Risk Level: {result.risk_level}")
        print(f"Estimated Duration: {result.estimated_duration}")
        print(f"Confidence: {result.confidence}%")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
