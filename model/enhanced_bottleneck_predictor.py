"""
Enhanced Bottleneck Predictor with Pilot Communications Integration
Uses real-time pilot communications transcription to validate and enhance bottleneck predictions
"""

import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import math
import re
from dataclasses import dataclass

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None
    print("Warning: cerebras-cloud-sdk not available. Bottleneck prediction will be disabled.")


@dataclass
class PilotCommunication:
    """Data structure for pilot communications"""
    timestamp: str
    frequency: str  # Radio frequency (e.g., "121.9", "Tower", "Ground")
    pilot_callsign: str  # Aircraft callsign
    message: str
    message_type: str  # "request", "instruction", "confirmation", "emergency"
    location_context: Optional[str] = None  # "runway", "taxiway", "gate", "approach"
    urgency_level: int = 1  # 1=normal, 2=urgent, 3=emergency


class TranscriptionEngine:
    """Mock transcription engine - replace with actual implementation"""
    
    def __init__(self):
        self.active_frequencies = {
            "121.9": "Tower",
            "121.7": "Ground", 
            "121.5": "Approach",
            "121.3": "Departure"
        }
        self.pilot_communications = []
    
    def transcribe_audio(self, audio_data: bytes, frequency: str) -> List[PilotCommunication]:
        """Transcribe audio data from a specific frequency"""
        # Mock implementation - replace with actual transcription service
        mock_transcriptions = [
            PilotCommunication(
                timestamp=datetime.now().isoformat(),
                frequency=frequency,
                pilot_callsign="DAL123",
                message="Tower, Delta 123, ready for departure runway 4L",
                message_type="request",
                location_context="runway",
                urgency_level=1
            ),
            PilotCommunication(
                timestamp=datetime.now().isoformat(),
                frequency=frequency,
                pilot_callsign="UAL456",
                message="Ground, United 456, taxiway Charlie, holding short",
                message_type="confirmation",
                location_context="taxiway",
                urgency_level=1
            )
        ]
        
        # Filter by frequency and add to communications log
        filtered_comms = [comm for comm in mock_transcriptions if comm.frequency == frequency]
        self.pilot_communications.extend(filtered_comms)
        
        return filtered_comms
    
    def get_recent_communications(self, minutes: int = 5) -> List[PilotCommunication]:
        """Get communications from the last N minutes"""
        cutoff_time = datetime.now().timestamp() - (minutes * 60)
        return [
            comm for comm in self.pilot_communications
            if datetime.fromisoformat(comm.timestamp).timestamp() > cutoff_time
        ]
    
    def analyze_communication_patterns(self) -> Dict:
        """Analyze communication patterns for bottleneck indicators"""
        recent_comms = self.get_recent_communications(10)  # Last 10 minutes
        
        if not recent_comms:
            return {"pattern_score": 0, "indicators": [], "urgency_level": 1}
        
        # Count different types of communications
        request_count = len([c for c in recent_comms if c.message_type == "request"])
        instruction_count = len([c for c in recent_comms if c.message_type == "instruction"])
        emergency_count = len([c for c in recent_comms if c.message_type == "emergency"])
        
        # Analyze location contexts
        runway_comms = len([c for c in recent_comms if c.location_context == "runway"])
        taxiway_comms = len([c for c in recent_comms if c.location_context == "taxiway"])
        
        # Calculate pattern score (0-100)
        pattern_score = 0
        
        # High request volume indicates congestion
        if request_count > 10:
            pattern_score += 30
        elif request_count > 5:
            pattern_score += 15
        
        # High instruction volume indicates active management
        if instruction_count > 8:
            pattern_score += 25
        elif instruction_count > 4:
            pattern_score += 12
        
        # Emergency communications are critical
        if emergency_count > 0:
            pattern_score += 50
        
        # Location-based analysis
        if runway_comms > taxiway_comms * 2:
            pattern_score += 20  # Runway congestion
        
        if taxiway_comms > runway_comms * 1.5:
            pattern_score += 15  # Taxiway congestion
        
        # Extract bottleneck indicators
        indicators = []
        for comm in recent_comms:
            message_lower = comm.message.lower()
            
            # Look for bottleneck-related keywords
            if any(word in message_lower for word in ["holding", "waiting", "delay", "congestion"]):
                indicators.append(f"Holding pattern detected: {comm.pilot_callsign}")
            
            if any(word in message_lower for word in ["runway", "departure", "takeoff"]):
                indicators.append(f"Runway activity: {comm.pilot_callsign}")
            
            if any(word in message_lower for word in ["taxi", "ground", "gate"]):
                indicators.append(f"Ground movement: {comm.pilot_callsign}")
        
        # Determine urgency level
        urgency_level = 1
        if emergency_count > 0:
            urgency_level = 3
        elif pattern_score > 60:
            urgency_level = 2
        
        return {
            "pattern_score": min(100, pattern_score),
            "indicators": indicators[:10],  # Top 10 indicators
            "urgency_level": urgency_level,
            "communication_count": len(recent_comms),
            "request_count": request_count,
            "instruction_count": instruction_count,
            "emergency_count": emergency_count,
            "runway_comms": runway_comms,
            "taxiway_comms": taxiway_comms
        }


class EnhancedBottleneckPredictor:
    def __init__(self):
        # Initialize Cerebras only if API key is available
        self.cerebras = None
        if Cerebras:
            try:
                import os
                if os.getenv('CEREBRAS_API_KEY'):
                    self.cerebras = Cerebras()
                    print("✅ Cerebras API initialized successfully")
                else:
                    print("⚠️  CEREBRAS_API_KEY not set - using simple prediction mode")
            except Exception as e:
                print(f"⚠️  Cerebras initialization failed: {e} - using simple prediction mode")
        
        self.results_file = "model/results.txt"
        self.transcription_engine = TranscriptionEngine()
        
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
    
    def correlate_communications_with_traffic(self, aircraft_data: List[Dict], communications: List[PilotCommunication]) -> Dict:
        """Correlate pilot communications with aircraft positions and movements"""
        correlation_analysis = {
            "matched_aircraft": 0,
            "communication_coverage": 0,
            "position_accuracy": 0,
            "movement_validation": 0,
            "discrepancies": []
        }
        
        if not communications or not aircraft_data:
            return correlation_analysis
        
        # Create aircraft lookup by callsign
        aircraft_lookup = {}
        for ac in aircraft_data:
            callsign = ac.get('flight', '').upper()
            aircraft_lookup[callsign] = ac
        
        # Match communications with aircraft
        matched_count = 0
        for comm in communications:
            callsign = comm.pilot_callsign.upper()
            if callsign in aircraft_lookup:
                matched_count += 1
                aircraft = aircraft_lookup[callsign]
                
                # Validate position context
                if comm.location_context == "runway" and aircraft.get('altitude') == 'ground':
                    correlation_analysis["position_accuracy"] += 1
                elif comm.location_context == "taxiway" and aircraft.get('altitude') == 'ground':
                    correlation_analysis["position_accuracy"] += 1
                elif comm.location_context == "approach" and isinstance(aircraft.get('altitude'), (int, float)) and aircraft.get('altitude', 0) < 5000:
                    correlation_analysis["position_accuracy"] += 1
                
                # Check for movement validation
                if "departure" in comm.message.lower() and aircraft.get('altitude') == 'ground':
                    correlation_analysis["movement_validation"] += 1
                elif "landing" in comm.message.lower() and isinstance(aircraft.get('altitude'), (int, float)) and aircraft.get('altitude', 0) < 3000:
                    correlation_analysis["movement_validation"] += 1
        
        correlation_analysis["matched_aircraft"] = matched_count
        correlation_analysis["communication_coverage"] = (matched_count / len(aircraft_data)) * 100 if aircraft_data else 0
        correlation_analysis["position_accuracy"] = (correlation_analysis["position_accuracy"] / max(matched_count, 1)) * 100
        correlation_analysis["movement_validation"] = (correlation_analysis["movement_validation"] / max(matched_count, 1)) * 100
        
        return correlation_analysis
    
    def predict_bottlenecks_with_cerebras(self, aircraft_data: List[Dict], airport_code: str, communications: List[PilotCommunication] = None) -> Dict:
        """Use Cerebras API to predict bottlenecks with communication context"""
        if not self.cerebras:
            return self.predict_bottlenecks_simple(aircraft_data, airport_code, communications)
        
        try:
            # Prepare data for Cerebras
            traffic_analysis = self.analyze_traffic_density(aircraft_data)
            comm_analysis = self.transcription_engine.analyze_communication_patterns()
            correlation_analysis = self.correlate_communications_with_traffic(aircraft_data, communications or [])
            
            prompt = f"""
            Analyze the following aircraft traffic data and pilot communications for {airport_code} airport to predict potential bottlenecks:
            
            Traffic Analysis:
            - Total aircraft: {len(aircraft_data)}
            - Ground aircraft: {traffic_analysis['ground_aircraft']}
            - Low altitude aircraft: {traffic_analysis['low_alt_aircraft']}
            - High altitude aircraft: {traffic_analysis['high_alt_aircraft']}
            - Density score: {traffic_analysis['density_score']}/100
            - Hotspots: {len(traffic_analysis['hotspots'])}
            
            Communication Analysis:
            - Pattern score: {comm_analysis['pattern_score']}/100
            - Communication count: {comm_analysis['communication_count']}
            - Request count: {comm_analysis['request_count']}
            - Instruction count: {comm_analysis['instruction_count']}
            - Emergency count: {comm_analysis['emergency_count']}
            - Urgency level: {comm_analysis['urgency_level']}
            
            Correlation Analysis:
            - Matched aircraft: {correlation_analysis['matched_aircraft']}
            - Communication coverage: {correlation_analysis['communication_coverage']:.1f}%
            - Position accuracy: {correlation_analysis['position_accuracy']:.1f}%
            - Movement validation: {correlation_analysis['movement_validation']:.1f}%
            
            Recent Communications:
            {json.dumps([{"callsign": c.pilot_callsign, "message": c.message, "type": c.message_type, "context": c.location_context} for c in (communications or [])[-5:]], indent=2)}
            
            Aircraft Details:
            {json.dumps(aircraft_data[:10], indent=2)}  # First 10 aircraft
            
            Predict:
            1. Likelihood of bottlenecks (0-100%)
            2. Primary bottleneck areas
            3. Communication-validated insights
            4. Recommended actions
            5. Risk level (Low/Medium/High)
            6. Confidence score based on data correlation
            """
            
            response = self.cerebras.generate(
                prompt=prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            return {
                "prediction_method": "cerebras_enhanced",
                "traffic_analysis": traffic_analysis,
                "communication_analysis": comm_analysis,
                "correlation_analysis": correlation_analysis,
                "cerebras_response": response,
                "timestamp": datetime.now().isoformat(),
                "airport": airport_code
            }
            
        except Exception as e:
            print(f"Error with Cerebras API: {e}")
            return self.predict_bottlenecks_simple(aircraft_data, airport_code, communications)
    
    def predict_bottlenecks_simple(self, aircraft_data: List[Dict], airport_code: str, communications: List[PilotCommunication] = None) -> Dict:
        """Enhanced simple bottleneck prediction with communication validation"""
        traffic_analysis = self.analyze_traffic_density(aircraft_data)
        comm_analysis = self.transcription_engine.analyze_communication_patterns()
        correlation_analysis = self.correlate_communications_with_traffic(aircraft_data, communications or [])
        
        # Enhanced bottleneck prediction logic
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
        
        # Communication pattern analysis
        if comm_analysis['pattern_score'] > 70:
            bottleneck_likelihood += 25
        elif comm_analysis['pattern_score'] > 40:
            bottleneck_likelihood += 15
        
        # Emergency communications
        if comm_analysis['emergency_count'] > 0:
            bottleneck_likelihood += 40
        
        # Density score contribution
        bottleneck_likelihood += min(25, traffic_analysis['density_score'] * 0.25)
        
        # Communication validation bonus/penalty
        if correlation_analysis['communication_coverage'] > 80:
            bottleneck_likelihood += 10  # High confidence due to good communication coverage
        elif correlation_analysis['communication_coverage'] < 30:
            bottleneck_likelihood -= 5  # Lower confidence due to poor communication coverage
        
        bottleneck_likelihood = min(100, max(0, bottleneck_likelihood))
        
        # Determine risk level
        if bottleneck_likelihood >= 70:
            risk_level = "High"
        elif bottleneck_likelihood >= 40:
            risk_level = "Medium"
        else:
            risk_level = "Low"
        
        return {
            "prediction_method": "simple_enhanced",
            "traffic_analysis": traffic_analysis,
            "communication_analysis": comm_analysis,
            "correlation_analysis": correlation_analysis,
            "bottleneck_likelihood": bottleneck_likelihood,
            "risk_level": risk_level,
            "confidence_score": self._calculate_confidence_score(traffic_analysis, comm_analysis, correlation_analysis),
            "recommendations": self._generate_enhanced_recommendations(bottleneck_likelihood, traffic_analysis, comm_analysis, correlation_analysis),
            "timestamp": datetime.now().isoformat(),
            "airport": airport_code
        }
    
    def _calculate_confidence_score(self, traffic_analysis: Dict, comm_analysis: Dict, correlation_analysis: Dict) -> float:
        """Calculate confidence score based on data quality and correlation"""
        confidence = 50  # Base confidence
        
        # Communication coverage bonus
        if correlation_analysis['communication_coverage'] > 80:
            confidence += 20
        elif correlation_analysis['communication_coverage'] > 50:
            confidence += 10
        
        # Position accuracy bonus
        if correlation_analysis['position_accuracy'] > 80:
            confidence += 15
        elif correlation_analysis['position_accuracy'] > 60:
            confidence += 8
        
        # Movement validation bonus
        if correlation_analysis['movement_validation'] > 70:
            confidence += 10
        elif correlation_analysis['movement_validation'] > 50:
            confidence += 5
        
        # Communication pattern consistency
        if comm_analysis['communication_count'] > 10:
            confidence += 5
        
        return min(100, confidence)
    
    def _generate_enhanced_recommendations(self, likelihood: float, traffic_analysis: Dict, comm_analysis: Dict, correlation_analysis: Dict) -> List[str]:
        """Generate enhanced recommendations based on all data sources"""
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
        
        # Communication-based recommendations
        if comm_analysis['emergency_count'] > 0:
            recommendations.append("🚨 EMERGENCY COMMUNICATIONS DETECTED - Immediate attention required")
        
        if comm_analysis['request_count'] > 10:
            recommendations.append("High communication volume - consider additional ATC support")
        
        if correlation_analysis['communication_coverage'] < 50:
            recommendations.append("Low communication coverage - enhance pilot communication monitoring")
        
        if correlation_analysis['position_accuracy'] < 70:
            recommendations.append("Position accuracy concerns - verify aircraft tracking systems")
        
        # Traffic-specific recommendations
        if traffic_analysis['ground_aircraft'] > 8:
            recommendations.append("High ground traffic - consider taxiway optimization")
        
        if len(traffic_analysis['hotspots']) > 2:
            recommendations.append("Multiple congestion areas detected - review routing")
        
        return recommendations
    
    def save_results(self, prediction_results: Dict):
        """Save enhanced prediction results to results.txt"""
        try:
            with open(self.results_file, 'a') as f:
                f.write(f"\n=== Enhanced Bottleneck Prediction - {prediction_results['timestamp']} ===\n")
                f.write(f"Airport: {prediction_results['airport']}\n")
                f.write(f"Method: {prediction_results['prediction_method']}\n")
                
                if 'bottleneck_likelihood' in prediction_results:
                    f.write(f"Bottleneck Likelihood: {prediction_results['bottleneck_likelihood']:.1f}%\n")
                    f.write(f"Risk Level: {prediction_results['risk_level']}\n")
                    f.write(f"Confidence Score: {prediction_results['confidence_score']:.1f}%\n")
                
                # Traffic Analysis
                traffic = prediction_results['traffic_analysis']
                f.write(f"\nTraffic Analysis:\n")
                f.write(f"  - Total Aircraft: {traffic['ground_aircraft'] + traffic['low_alt_aircraft'] + traffic['high_alt_aircraft']}\n")
                f.write(f"  - Ground: {traffic['ground_aircraft']}\n")
                f.write(f"  - Low Altitude: {traffic['low_alt_aircraft']}\n")
                f.write(f"  - High Altitude: {traffic['high_alt_aircraft']}\n")
                f.write(f"  - Density Score: {traffic['density_score']:.1f}/100\n")
                f.write(f"  - Hotspots: {len(traffic['hotspots'])}\n")
                
                # Communication Analysis
                if 'communication_analysis' in prediction_results:
                    comm = prediction_results['communication_analysis']
                    f.write(f"\nCommunication Analysis:\n")
                    f.write(f"  - Pattern Score: {comm['pattern_score']:.1f}/100\n")
                    f.write(f"  - Communication Count: {comm['communication_count']}\n")
                    f.write(f"  - Request Count: {comm['request_count']}\n")
                    f.write(f"  - Instruction Count: {comm['instruction_count']}\n")
                    f.write(f"  - Emergency Count: {comm['emergency_count']}\n")
                    f.write(f"  - Urgency Level: {comm['urgency_level']}\n")
                    
                    if comm['indicators']:
                        f.write(f"  - Key Indicators:\n")
                        for indicator in comm['indicators'][:5]:
                            f.write(f"    * {indicator}\n")
                
                # Correlation Analysis
                if 'correlation_analysis' in prediction_results:
                    corr = prediction_results['correlation_analysis']
                    f.write(f"\nCorrelation Analysis:\n")
                    f.write(f"  - Matched Aircraft: {corr['matched_aircraft']}\n")
                    f.write(f"  - Communication Coverage: {corr['communication_coverage']:.1f}%\n")
                    f.write(f"  - Position Accuracy: {corr['position_accuracy']:.1f}%\n")
                    f.write(f"  - Movement Validation: {corr['movement_validation']:.1f}%\n")
                
                # Recommendations
                if 'recommendations' in prediction_results:
                    f.write(f"\nRecommendations:\n")
                    for rec in prediction_results['recommendations']:
                        f.write(f"  - {rec}\n")
                
                # Cerebras Analysis
                if 'cerebras_response' in prediction_results:
                    f.write(f"\nCerebras Analysis:\n{prediction_results['cerebras_response']}\n")
                
                f.write("\n" + "="*60 + "\n")
                
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def predict_and_save(self, aircraft_data: List[Dict], airport_code: str, communications: List[PilotCommunication] = None) -> Dict:
        """Main method to predict bottlenecks with communication validation and save results"""
        prediction_results = self.predict_bottlenecks_with_cerebras(aircraft_data, airport_code, communications)
        self.save_results(prediction_results)
        return prediction_results
    
    def simulate_pilot_communications(self, aircraft_data: List[Dict]) -> List[PilotCommunication]:
        """Simulate pilot communications based on aircraft data (for testing)"""
        communications = []
        
        for ac in aircraft_data[:5]:  # Simulate communications for first 5 aircraft
            callsign = ac.get('flight', 'Unknown')
            altitude = ac.get('altitude')
            
            if altitude == 'ground':
                comm = PilotCommunication(
                    timestamp=datetime.now().isoformat(),
                    frequency="121.7",
                    pilot_callsign=callsign,
                    message=f"Ground, {callsign}, ready for pushback",
                    message_type="request",
                    location_context="gate",
                    urgency_level=1
                )
            elif isinstance(altitude, (int, float)) and altitude < 3000:
                comm = PilotCommunication(
                    timestamp=datetime.now().isoformat(),
                    frequency="121.9",
                    pilot_callsign=callsign,
                    message=f"Tower, {callsign}, on final approach",
                    message_type="confirmation",
                    location_context="approach",
                    urgency_level=1
                )
            else:
                comm = PilotCommunication(
                    timestamp=datetime.now().isoformat(),
                    frequency="121.5",
                    pilot_callsign=callsign,
                    message=f"Approach, {callsign}, requesting descent",
                    message_type="request",
                    location_context="approach",
                    urgency_level=1
                )
            
            communications.append(comm)
        
        return communications


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
    
    predictor = EnhancedBottleneckPredictor()
    
    # Simulate pilot communications
    communications = predictor.simulate_pilot_communications(sample_aircraft)
    
    # Predict bottlenecks with communication validation
    results = predictor.predict_and_save(sample_aircraft, "JFK", communications)
    print("Enhanced prediction completed and saved to results.txt")
