"""
Simple Bottleneck Predictor using Cerebras API
Takes flight lat/lon and trajectories to predict potential bottlenecks
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Tuple
import math

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None
    print("Warning: cerebras-cloud-sdk not available. Bottleneck prediction will be disabled.")


class SimpleBottleneckPredictor:
    def __init__(self):
        self.cerebras = Cerebras() if Cerebras else None
        self.results_file = "model/results.txt"
        
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
    
    def analyze_traffic_density(self, aircraft_data: List[Dict]) -> Dict:
        """Analyze traffic density patterns"""
        if not aircraft_data:
            return {"density_score": 0, "hotspots": [], "congestion_areas": []}
        
        # Group aircraft by altitude bands
        ground_aircraft = [ac for ac in aircraft_data if ac.get('altitude') == 'ground']
        low_alt_aircraft = [ac for ac in aircraft_data if isinstance(ac.get('altitude'), (int, float)) and ac.get('altitude', 0) < 5000]
        high_alt_aircraft = [ac for ac in aircraft_data if isinstance(ac.get('altitude'), (int, float)) and ac.get('altitude', 0) >= 5000]
        
        # Calculate density metrics
        total_aircraft = len(aircraft_data)
        ground_density = len(ground_aircraft) / max(total_aircraft, 1)
        low_alt_density = len(low_alt_aircraft) / max(total_aircraft, 1)
        
        # Find hotspots (areas with high aircraft concentration)
        hotspots = []
        if len(aircraft_data) > 1:
            for i, ac1 in enumerate(aircraft_data):
                nearby_count = 0
                for j, ac2 in enumerate(aircraft_data):
                    if i != j:
                        dist = self.calculate_distance(
                            ac1.get('lat', 0), ac1.get('lon', 0),
                            ac2.get('lat', 0), ac2.get('lon', 0)
                        )
                        if dist < 5:  # Within 5 nautical miles
                            nearby_count += 1
                
                if nearby_count >= 3:  # High concentration area
                    hotspots.append({
                        'lat': ac1.get('lat'),
                        'lon': ac1.get('lon'),
                        'aircraft_count': nearby_count + 1,
                        'flight': ac1.get('flight', 'Unknown')
                    })
        
        # Calculate overall density score (0-100)
        density_score = min(100, (ground_density * 50) + (low_alt_density * 30) + (len(hotspots) * 5))
        
        return {
            "density_score": density_score,
            "hotspots": hotspots[:5],  # Top 5 hotspots
            "congestion_areas": [h for h in hotspots if h['aircraft_count'] >= 5],
            "ground_aircraft": len(ground_aircraft),
            "low_alt_aircraft": len(low_alt_aircraft),
            "high_alt_aircraft": len(high_alt_aircraft)
        }
    
    def predict_bottlenecks_with_cerebras(self, aircraft_data: List[Dict], airport_code: str) -> Dict:
        """Use Cerebras API to predict bottlenecks"""
        if not self.cerebras:
            return self.predict_bottlenecks_simple(aircraft_data, airport_code)
        
        try:
            # Prepare data for Cerebras
            traffic_analysis = self.analyze_traffic_density(aircraft_data)
            
            prompt = f"""
            Analyze the following aircraft traffic data for {airport_code} airport and predict potential bottlenecks:
            
            Traffic Analysis:
            - Total aircraft: {len(aircraft_data)}
            - Ground aircraft: {traffic_analysis['ground_aircraft']}
            - Low altitude aircraft: {traffic_analysis['low_alt_aircraft']}
            - High altitude aircraft: {traffic_analysis['high_alt_aircraft']}
            - Density score: {traffic_analysis['density_score']}/100
            - Hotspots: {len(traffic_analysis['hotspots'])}
            
            Aircraft Details:
            {json.dumps(aircraft_data[:10], indent=2)}  # First 10 aircraft
            
            Predict:
            1. Likelihood of bottlenecks (0-100%)
            2. Primary bottleneck areas
            3. Recommended actions
            4. Risk level (Low/Medium/High)
            """
            
            response = self.cerebras.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            
            return {
                "prediction_method": "cerebras",
                "traffic_analysis": traffic_analysis,
                "cerebras_response": response,
                "timestamp": datetime.now().isoformat(),
                "airport": airport_code
            }
            
        except Exception as e:
            print(f"Error with Cerebras API: {e}")
            return self.predict_bottlenecks_simple(aircraft_data, airport_code)
    
    def predict_bottlenecks_simple(self, aircraft_data: List[Dict], airport_code: str) -> Dict:
        """Simple bottleneck prediction without Cerebras"""
        traffic_analysis = self.analyze_traffic_density(aircraft_data)
        
        # Simple bottleneck prediction logic
        bottleneck_likelihood = 0
        
        # Ground congestion
        if traffic_analysis['ground_aircraft'] > 10:
            bottleneck_likelihood += 30
        elif traffic_analysis['ground_aircraft'] > 5:
            bottleneck_likelihood += 15
        
        # Low altitude congestion
        if traffic_analysis['low_alt_aircraft'] > 15:
            bottleneck_likelihood += 25
        elif traffic_analysis['low_alt_aircraft'] > 8:
            bottleneck_likelihood += 12
        
        # Hotspot analysis
        if len(traffic_analysis['hotspots']) > 3:
            bottleneck_likelihood += 20
        elif len(traffic_analysis['hotspots']) > 1:
            bottleneck_likelihood += 10
        
        # Density score contribution
        bottleneck_likelihood += min(25, traffic_analysis['density_score'] * 0.25)
        
        bottleneck_likelihood = min(100, bottleneck_likelihood)
        
        # Determine risk level
        if bottleneck_likelihood >= 70:
            risk_level = "High"
        elif bottleneck_likelihood >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "prediction_method": "simple",
            "traffic_analysis": traffic_analysis,
            "bottleneck_likelihood": bottleneck_likelihood,
            "risk_level": risk_level,
            "recommendations": self._generate_recommendations(bottleneck_likelihood, traffic_analysis),
            "timestamp": datetime.now().isoformat(),
            "airport": airport_code
        }
    
    def _generate_recommendations(self, likelihood: float, traffic_analysis: Dict) -> List[str]:
        """Generate recommendations based on bottleneck likelihood"""
        recommendations = []
        
        if likelihood >= 70:
            recommendations.extend([
                "Consider implementing ground stop procedures",
                "Increase spacing between departing aircraft",
                "Monitor taxiway congestion closely",
                "Prepare for potential delays"
            ])
        elif likelihood >= 40:
            recommendations.extend([
                "Monitor traffic density closely",
                "Consider minor spacing adjustments",
                "Prepare contingency plans"
            ])
        else:
            recommendations.extend([
                "Traffic flow appears normal",
                "Continue standard operations",
                "Monitor for changes"
            ])
        
        if traffic_analysis['ground_aircraft'] > 8:
            recommendations.append("High ground traffic - consider taxiway optimization")
        
        if len(traffic_analysis['hotspots']) > 2:
            recommendations.append("Multiple congestion areas detected - review routing")
        
        return recommendations
    
    def save_results(self, prediction_results: Dict):
        """Save prediction results to results.txt"""
        try:
            with open(self.results_file, 'a') as f:
                f.write(f"\n=== Bottleneck Prediction - {prediction_results['timestamp']} ===\n")
                f.write(f"Airport: {prediction_results['airport']}\n")
                f.write(f"Method: {prediction_results['prediction_method']}\n")
                
                if 'bottleneck_likelihood' in prediction_results:
                    f.write(f"Bottleneck Likelihood: {prediction_results['bottleneck_likelihood']:.1f}%\n")
                    f.write(f"Risk Level: {prediction_results['risk_level']}\n")
                
                traffic = prediction_results['traffic_analysis']
                f.write(f"Traffic Analysis:\n")
                f.write(f"  - Total Aircraft: {traffic['ground_aircraft'] + traffic['low_alt_aircraft'] + traffic['high_alt_aircraft']}\n")
                f.write(f"  - Ground: {traffic['ground_aircraft']}\n")
                f.write(f"  - Low Altitude: {traffic['low_alt_aircraft']}\n")
                f.write(f"  - High Altitude: {traffic['high_alt_aircraft']}\n")
                f.write(f"  - Density Score: {traffic['density_score']:.1f}/100\n")
                f.write(f"  - Hotspots: {len(traffic['hotspots'])}\n")
                
                if 'recommendations' in prediction_results:
                    f.write(f"Recommendations:\n")
                    for rec in prediction_results['recommendations']:
                        f.write(f"  - {rec}\n")
                
                if 'cerebras_response' in prediction_results:
                    f.write(f"Cerebras Analysis:\n{prediction_results['cerebras_response']}\n")
                
                f.write("\n" + "="*60 + "\n")
                
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def predict_and_save(self, aircraft_data: List[Dict], airport_code: str) -> Dict:
        """Main method to predict bottlenecks and save results"""
        prediction_results = self.predict_bottlenecks_with_cerebras(aircraft_data, airport_code)
        self.save_results(prediction_results)
        return prediction_results


# Example usage
if __name__ == "__main__":
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
        }
    ]
    
    predictor = SimpleBottleneckPredictor()
    results = predictor.predict_and_save(sample_aircraft, "JFK")
    print("Prediction completed and saved to results.txt")
