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
    """Structured bottleneck prediction result"""
    severity_score: float  # 0-100
    risk_level: str  # "Low", "Medium", "High", "Critical"
    bottleneck_locations: List[str]
    estimated_duration: str
    affected_aircraft: List[str]
    recommendations: List[str]
    confidence: float
    timestamp: str


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
        # Load API key from environment or gemini_config.env file
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        # If not found in environment, try to load from gemini_config.env file
        if not self.api_key:
            # Try multiple possible locations for the config file
            config_paths = [
                'gemini_config.env',  # Current directory
                '../gemini_config.env',  # Parent directory (when called from model/)
            ]
            
            # Add relative path from this file if __file__ is available
            try:
                config_paths.append(os.path.join(os.path.dirname(__file__), '..', 'gemini_config.env'))
            except NameError:
                pass  # __file__ not available in some contexts
            
            for config_path in config_paths:
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            for line in f:
                                if line.startswith('GEMINI_API_KEY='):
                                    self.api_key = line.split('=', 1)[1].strip()
                                    print(f"✅ Gemini API key loaded from {config_path}")
                                    break
                        if self.api_key:
                            break
                    except Exception as e:
                        print(f"⚠️ Failed to load Gemini API key from {config_path}: {e}")
        
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set or available in gemini_config.env file")
        
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        self.results_file = "model/results.txt"
        print("✅ Gemini Bottleneck Predictor initialized successfully")
    
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
        
        # Calculate density score
        total_aircraft = len(aircraft_data)
        ground_ratio = len(ground_aircraft) / max(total_aircraft, 1)
        hotspot_factor = len(hotspots) * 10
        
        density_score = min(100, (ground_ratio * 60) + hotspot_factor)
        
        # Identify congestion areas
        congestion_areas = [h for h in hotspots if h['aircraft_count'] >= 4]
        
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
                    if dist < 3:  # Within 3 nautical miles
                        nearby_aircraft.append(ac2)
            
            if len(nearby_aircraft) >= 2:  # At least 3 aircraft total
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

REQUIRED OUTPUT FORMAT (JSON only, no extra text):

{{
  "bottleneck_id": "taxiway_2025-09-21_01:21:42",
  "coordinates": [40.6398473, -73.7773369],
  "timestamp": "2025-09-21 01:21:42",
  "type": "taxiway",
  "severity": 4,
  "duration": 20.0,
  "confidence": 0.6,
  "aircraft_count": 10,
  "aircraft_affected": [
    {{
      "flight_id": "DAL123",
      "aircraft_type": "B737",
      "position": [40.6413, -73.7781],
      "time": "2025-09-21T01:21:42"
    }}
  ]
}}

Notes:
- `bottleneck_id` should combine bottleneck type + timestamp.
- `coordinates` is the average [lat, lon] of affected aircraft.
- `timestamp` is the current time (YYYY-MM-DD HH:MM:SS).
- `type` can be "runway", "taxiway", "terminal", or "unknown".
- `severity` is an integer 1–5 (low to high).
- `duration` is in minutes (float).
- `confidence` is a probability 0.0–1.0.
- `aircraft_count` is total aircraft involved.
- `aircraft_affected` lists up to 10 aircraft with id, type, position, and timestamp.

Provide your analysis strictly in this JSON format, without commentary.
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
            print(":robot_face: Running Gemini AI analysis...")
            response = await self._get_gemini_response(prompt)
            return response
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
                print(f":warning: Failed to parse Gemini response as JSON: {e}")
                return self._fallback_prediction(traffic_analysis, airport_code, aircraft_data)
        except Exception as e:
            print(f":warning: Gemini analysis failed: {e}")
            return self._fallback_prediction(traffic_analysis, airport_code, aircraft_data)
    
    async def _get_gemini_response(self, prompt: str) -> str:
        """Get response from Gemini API with retry logic"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                if response and response.text:
                    return response.text
                else:
                    print(f"⚠️ Gemini API returned empty response on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        raise Exception("Gemini API returned empty response after all retries")
            except Exception as e:
                print(f"⚠️ Gemini API attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    raise e
    
    def _fallback_prediction(self, traffic_analysis: TrafficAnalysis, airport_code: str) -> BottleneckPrediction:
        """Fallback prediction when Gemini is unavailable"""
        
        # Simple heuristic-based prediction
        severity_score = min(100, traffic_analysis.density_score)
        
        if severity_score >= 80:
            risk_level = "Critical"
            duration = "60+ minutes"
            recommendations = [
                "Implement immediate ground stop",
                "Activate emergency procedures",
                "Contact airport management"
            ]
        elif severity_score >= 60:
            risk_level = "High"
            duration = "30-45 minutes"
            recommendations = [
                "Increase aircraft spacing",
                "Monitor taxiway congestion",
                "Prepare for delays"
            ]
        elif severity_score >= 40:
            risk_level = "Medium"
            duration = "15-30 minutes"
            recommendations = [
                "Monitor traffic density",
                "Consider minor adjustments",
                "Prepare contingency plans"
            ]
        else:
            risk_level = "Low"
            duration = "5-15 minutes"
            recommendations = [
                "Continue normal operations",
                "Monitor for changes"
            ]
        
        return BottleneckPrediction(
            severity_score=severity_score,
            risk_level=risk_level,
            bottleneck_locations=["General airport area"] if severity_score > 50 else [],
            estimated_duration=duration,
            affected_aircraft=[f"Aircraft-{i}" for i in range(min(5, traffic_analysis.total_aircraft))],
            recommendations=recommendations,
            confidence=60,  # Lower confidence for fallback
            timestamp=datetime.now().isoformat()
        )
    
    def save_results(self, prediction: BottleneckPrediction, traffic_analysis: TrafficAnalysis, airport_code: str):
        """Save prediction results to results.txt"""
        try:
            with open(self.results_file, 'a') as f:
                f.write(f"\n=== Gemini Bottleneck Prediction - {prediction.timestamp} ===\n")
                f.write(f"Airport: {airport_code}\n")
                f.write(f"Severity Score: {prediction.severity_score:.1f}/100\n")
                f.write(f"Risk Level: {prediction.risk_level}\n")
                f.write(f"Estimated Duration: {prediction.estimated_duration}\n")
                f.write(f"Confidence: {prediction.confidence:.1f}%\n")
                
                f.write(f"\nTraffic Analysis:\n")
                f.write(f"  - Total Aircraft: {traffic_analysis.total_aircraft}\n")
                f.write(f"  - Ground Aircraft: {traffic_analysis.ground_aircraft}\n")
                f.write(f"  - Airborne Aircraft: {traffic_analysis.airborne_aircraft}\n")
                f.write(f"  - Density Score: {traffic_analysis.density_score:.1f}/100\n")
                f.write(f"  - Hotspots: {len(traffic_analysis.hotspots)}\n")
                f.write(f"  - Congestion Areas: {len(traffic_analysis.congestion_areas)}\n")
                
                if prediction.bottleneck_locations:
                    f.write(f"\nBottleneck Locations:\n")
                    for location in prediction.bottleneck_locations:
                        f.write(f"  - {location}\n")
                
                if prediction.affected_aircraft:
                    f.write(f"\nAffected Aircraft:\n")
                    for aircraft in prediction.affected_aircraft:
                        f.write(f"  - {aircraft}\n")
                
                if prediction.recommendations:
                    f.write(f"\nRecommendations:\n")
                    for rec in prediction.recommendations:
                        f.write(f"  - {rec}\n")
                
                f.write("\n" + "="*60 + "\n")
                
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def create_bottleneck_card(self, prediction_results) -> Dict:
        """Create a structured bottleneck card from prediction results"""
        import uuid
        from datetime import datetime
        
        # Extract data from prediction results
        if isinstance(prediction_results, BottleneckPrediction):
            severity_score = prediction_results.severity_score
            risk_level = prediction_results.risk_level
            bottleneck_locations = prediction_results.bottleneck_locations
            estimated_duration = prediction_results.estimated_duration
            affected_aircraft = prediction_results.affected_aircraft
            recommendations = prediction_results.recommendations
            confidence = prediction_results.confidence
        else:
            # Handle dict format
            severity_score = prediction_results.get('severity_score', 0)
            risk_level = prediction_results.get('risk_level', 'Low')
            bottleneck_locations = prediction_results.get('bottleneck_locations', [])
            estimated_duration = prediction_results.get('estimated_duration', 'Unknown')
            affected_aircraft = prediction_results.get('affected_aircraft', [])
            recommendations = prediction_results.get('recommendations', [])
            confidence = prediction_results.get('confidence', 0)
        
        # Convert severity score to 1-5 scale
        severity_level = max(1, min(5, round(severity_score / 20)))
        
        # Estimate delay minutes from duration string
        estimated_delay_minutes = 30  # Default
        if 'minutes' in estimated_duration.lower():
            try:
                estimated_delay_minutes = int(''.join(filter(str.isdigit, estimated_duration)))
            except:
                estimated_delay_minutes = 30
        
        # Create bottleneck card
        bottleneck_card = {
            "bottleneck_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "airport_code": "JFK",  # Default, should be passed as parameter
            "type": "traffic_congestion",
            "severity": severity_level,
            "risk_level": risk_level,
            "locations": bottleneck_locations,
            "impact": {
                "estimated_delay_minutes": estimated_delay_minutes,
                "aircraft_affected": len(affected_aircraft),
                "confidence": confidence,
                "economic_impact_usd": estimated_delay_minutes * len(affected_aircraft) * 150,  # Rough estimate
                "passengers_affected": len(affected_aircraft) * 150  # Average passengers per aircraft
            },
            "recommendations": recommendations,
            "status": "active"
        }
        
        return bottleneck_card

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