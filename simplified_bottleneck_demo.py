"""
Simplified Bottleneck Prediction Demo

This script demonstrates bottleneck prediction using sample aircraft data
without requiring the full ML model dependencies.
"""

import json
import math
from datetime import datetime
from typing import Dict, List, Tuple


class SimplifiedBottleneckPredictor:
    """
    Simplified bottleneck prediction system that demonstrates the concept
    without requiring PyTorch or other ML dependencies
    """
    
    def __init__(self):
        # Aircraft database
        self.aircraft_database = {
            'B737': {'passengers': 150, 'fuel_burn_idle': 150, 'category': 'narrow_body'},
            'A320': {'passengers': 180, 'fuel_burn_idle': 140, 'category': 'narrow_body'},
            'B777': {'passengers': 350, 'fuel_burn_idle': 300, 'category': 'wide_body'},
            'A380': {'passengers': 550, 'fuel_burn_idle': 500, 'category': 'wide_body'},
            'B767F': {'passengers': 0, 'fuel_burn_idle': 250, 'category': 'cargo'},
        }
        
        # Airport zones (simplified)
        self.airport_zones = {
            'runway_approach': {'capacity': 3, 'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.001},
            'runway_departure': {'capacity': 2, 'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.0008},
            'taxiway_intersection': {'capacity': 1, 'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.0005},
            'gate_area': {'capacity': 8, 'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.002}
        }
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in degrees"""
        return math.sqrt((lat1 - lat2)**2 + (lon1 - lon2)**2)
    
    def determine_aircraft_phase(self, aircraft: Dict) -> str:
        """Determine aircraft operational phase"""
        altitude = aircraft.get('alt_baro', 0)
        speed = aircraft.get('gs', 0)
        
        if altitude > 3000:
            return 'approach' if speed < 200 else 'departure'
        elif altitude < 100:
            return 'gate' if speed < 10 else 'taxi'
        else:
            return 'taxi'
    
    def estimate_passenger_count(self, aircraft_type: str) -> int:
        """Estimate passenger count based on aircraft type"""
        specs = self.aircraft_database.get(aircraft_type, self.aircraft_database['B737'])
        base_passengers = specs['passengers']
        
        # Apply load factor (80% average)
        return int(base_passengers * 0.8)
    
    def identify_bottleneck_zones(self, aircraft_list: List[Dict]) -> List[Dict]:
        """Identify bottleneck zones based on aircraft density"""
        bottlenecks = []
        
        for zone_name, zone_config in self.airport_zones.items():
            zone_aircraft = []
            
            # Find aircraft in this zone
            for aircraft in aircraft_list:
                lat = aircraft.get('lat', 0)
                lon = aircraft.get('lon', 0)
                
                distance = self.calculate_distance(
                    lat, lon, 
                    zone_config['center_lat'], 
                    zone_config['center_lon']
                )
                
                if distance <= zone_config['radius']:
                    zone_aircraft.append(aircraft)
            
            # Calculate bottleneck probability
            capacity = zone_config['capacity']
            current_load = len(zone_aircraft)
            
            if current_load > 0:
                utilization = current_load / capacity
                bottleneck_prob = min(utilization, 1.0)
                
                if bottleneck_prob > 0.6:  # Threshold for bottleneck
                    bottlenecks.append({
                        'zone_id': zone_name,
                        'zone_type': zone_name,
                        'center_lat': zone_config['center_lat'],
                        'center_lon': zone_config['center_lon'],
                        'radius_meters': zone_config['radius'] * 111000,  # Convert to meters
                        'capacity': capacity,
                        'current_load': current_load,
                        'bottleneck_probability': bottleneck_prob,
                        'aircraft_affected': zone_aircraft
                    })
        
        return bottlenecks
    
    def calculate_impact_metrics(self, bottlenecks: List[Dict]) -> Dict:
        """Calculate impact metrics for bottlenecks"""
        total_passengers = 0
        total_fuel_waste = 0
        total_economic_impact = 0
        total_co2_emissions = 0
        
        for bottleneck in bottlenecks:
            zone_aircraft = bottleneck['aircraft_affected']
            delay_minutes = 15  # Assume 15 minute delay
            
            for aircraft in zone_aircraft:
                aircraft_type = aircraft.get('t', 'B737')
                specs = self.aircraft_database.get(aircraft_type, self.aircraft_database['B737'])
                
                # Passenger impact
                passengers = self.estimate_passenger_count(aircraft_type)
                total_passengers += passengers
                
                # Fuel waste
                fuel_burn_rate = specs['fuel_burn_idle']
                fuel_waste = (fuel_burn_rate / 60) * delay_minutes
                total_fuel_waste += fuel_waste
                
                # Economic impact
                fuel_cost = fuel_waste * 3.50  # $3.50 per gallon
                passenger_compensation = passengers * delay_minutes * 2.50  # $2.50 per passenger per minute
                economic_impact = fuel_cost + passenger_compensation
                total_economic_impact += economic_impact
                
                # CO2 emissions
                co2_emissions = fuel_waste * 21.1  # lbs CO2 per gallon
                total_co2_emissions += co2_emissions
        
        return {
            'total_passengers_affected': total_passengers,
            'total_fuel_waste_gallons': total_fuel_waste,
            'total_economic_impact_usd': total_economic_impact,
            'total_co2_emissions_lbs': total_co2_emissions,
            'average_delay_minutes': 15
        }
    
    def generate_mitigation_recommendations(self, bottleneck: Dict) -> List[Dict]:
        """Generate mitigation recommendations for bottlenecks"""
        zone_type = bottleneck['zone_type']
        severity = bottleneck['bottleneck_probability']
        
        recommendations = []
        
        if zone_type == 'runway_approach':
            recommendations.extend([
                {
                    'action': 'Increase runway separation',
                    'priority': 'high' if severity > 0.8 else 'medium',
                    'estimated_effectiveness': 0.8,
                    'implementation_time': 5.0
                },
                {
                    'action': 'Divert aircraft to alternate runways',
                    'priority': 'high' if severity > 0.9 else 'medium',
                    'estimated_effectiveness': 0.9,
                    'implementation_time': 10.0
                }
            ])
        elif zone_type == 'runway_departure':
            recommendations.extend([
                {
                    'action': 'Optimize departure sequencing',
                    'priority': 'medium',
                    'estimated_effectiveness': 0.6,
                    'implementation_time': 3.0
                },
                {
                    'action': 'Use multiple departure runways',
                    'priority': 'high' if severity > 0.7 else 'medium',
                    'estimated_effectiveness': 0.8,
                    'implementation_time': 8.0
                }
            ])
        elif zone_type == 'taxiway_intersection':
            recommendations.extend([
                {
                    'action': 'Implement ground traffic sequencing',
                    'priority': 'medium',
                    'estimated_effectiveness': 0.7,
                    'implementation_time': 3.0
                },
                {
                    'action': 'Use alternative taxi routes',
                    'priority': 'low',
                    'estimated_effectiveness': 0.5,
                    'implementation_time': 2.0
                }
            ])
        elif zone_type == 'gate_area':
            recommendations.extend([
                {
                    'action': 'Reassign gates dynamically',
                    'priority': 'high' if severity > 0.6 else 'medium',
                    'estimated_effectiveness': 0.8,
                    'implementation_time': 5.0
                },
                {
                    'action': 'Use remote parking positions',
                    'priority': 'medium',
                    'estimated_effectiveness': 0.6,
                    'implementation_time': 8.0
                }
            ])
        
        return recommendations
    
    def predict_bottlenecks(self, adsb_data: Dict, airport_config: Dict) -> Dict:
        """Main prediction function"""
        aircraft_list = adsb_data.get('aircraft', [])
        
        if not aircraft_list:
            return self._create_empty_response()
        
        # Identify bottlenecks
        bottlenecks = self.identify_bottleneck_zones(aircraft_list)
        
        # Calculate impact metrics
        impact_metrics = self.calculate_impact_metrics(bottlenecks)
        
        # Generate predictions
        bottleneck_predictions = []
        
        for bottleneck in bottlenecks:
            # Generate mitigation recommendations
            recommendations = self.generate_mitigation_recommendations(bottleneck)
            
            # Calculate individual bottleneck impact
            zone_aircraft = bottleneck['aircraft_affected']
            zone_passengers = sum(self.estimate_passenger_count(ac.get('t', 'B737')) for ac in zone_aircraft)
            zone_fuel_waste = sum((self.aircraft_database.get(ac.get('t', 'B737'), self.aircraft_database['B737'])['fuel_burn_idle'] / 60) * 15 for ac in zone_aircraft)
            
            prediction = {
                'bottleneck_id': bottleneck['zone_id'],
                'location': {
                    'zone': bottleneck['zone_type'],
                    'coordinates': [bottleneck['center_lat'], bottleneck['center_lon']]
                },
                'type': bottleneck['zone_type'],
                'probability': bottleneck['bottleneck_probability'],
                'severity': int(bottleneck['bottleneck_probability'] * 5) + 1,
                
                'timing': {
                    'predicted_onset_minutes': 0,
                    'estimated_duration_minutes': 15.0,
                    'resolution_confidence': 1.0 - bottleneck['bottleneck_probability']
                },
                
                'aircraft_affected': [
                    {
                        'flight_id': ac.get('flight', 'UNKNOWN'),
                        'aircraft_type': ac.get('t', 'UNKNOWN'),
                        'estimated_passengers': self.estimate_passenger_count(ac.get('t', 'B737')),
                        'delay_contribution': bottleneck['bottleneck_probability'] * 10,
                        'current_phase': self.determine_aircraft_phase(ac)
                    }
                    for ac in zone_aircraft
                ],
                
                'impact_analysis': {
                    'passengers_affected': zone_passengers,
                    'total_delay_minutes': 15.0,
                    'fuel_waste_gallons': zone_fuel_waste,
                    'fuel_cost_estimate': zone_fuel_waste * 3.50,
                    'co2_emissions_lbs': zone_fuel_waste * 21.1,
                    'economic_impact_estimate': (zone_fuel_waste * 3.50) + (zone_passengers * 15 * 2.50)
                },
                
                'recommended_mitigations': recommendations
            }
            
            bottleneck_predictions.append(prediction)
        
        # Create airport summary
        highest_severity = max([b['bottleneck_probability'] for b in bottlenecks]) if bottlenecks else 0
        overall_risk = 'critical' if highest_severity > 0.9 else 'high' if highest_severity > 0.7 else 'medium' if highest_severity > 0.5 else 'low'
        
        airport_summary = {
            'total_bottlenecks_predicted': len(bottlenecks),
            'highest_severity_level': int(highest_severity * 5) + 1,
            'total_passengers_at_risk': impact_metrics['total_passengers_affected'],
            'total_fuel_waste_estimate': impact_metrics['total_fuel_waste_gallons'],
            'overall_delay_risk': overall_risk
        }
        
        return {
            'timestamp': datetime.now().isoformat(),
            'airport': airport_config.get('icao', 'KJFK'),
            'analysis_radius_nm': 3,
            'total_aircraft_monitored': len(aircraft_list),
            'bottleneck_predictions': bottleneck_predictions,
            'airport_summary': airport_summary,
            'impact_metrics': impact_metrics
        }
    
    def _create_empty_response(self) -> Dict:
        """Create empty response when no data"""
        return {
            'timestamp': datetime.now().isoformat(),
            'airport': 'KJFK',
            'analysis_radius_nm': 3,
            'total_aircraft_monitored': 0,
            'bottleneck_predictions': [],
            'airport_summary': {
                'total_bottlenecks_predicted': 0,
                'highest_severity_level': 1,
                'total_passengers_at_risk': 0,
                'total_fuel_waste_estimate': 0,
                'overall_delay_risk': 'low'
            }
        }


def load_sample_data():
    """Load sample aircraft data"""
    try:
        with open('sample_aircraft_data.json', 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded sample data: {len(data.get('aircraft', []))} aircraft")
        return data
    except FileNotFoundError:
        print("❌ sample_aircraft_data.json not found")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Error parsing JSON: {e}")
        return None


def display_results(analysis):
    """Display analysis results"""
    print(f"\n📊 BOTTLENECK ANALYSIS RESULTS")
    print("=" * 50)
    
    # Airport summary
    summary = analysis.get('airport_summary', {})
    print(f"Airport: {analysis.get('airport', 'UNKNOWN')}")
    print(f"Aircraft monitored: {analysis.get('total_aircraft_monitored', 0)}")
    print(f"Bottlenecks predicted: {summary.get('total_bottlenecks_predicted', 0)}")
    print(f"Highest severity: {summary.get('highest_severity_level', 1)}/5")
    print(f"Overall delay risk: {summary.get('overall_delay_risk', 'low').upper()}")
    print(f"Passengers at risk: {summary.get('total_passengers_at_risk', 0)}")
    print(f"Fuel waste estimate: {summary.get('total_fuel_waste_estimate', 0):.1f} gallons")
    
    # Impact metrics
    impact = analysis.get('impact_metrics', {})
    if impact:
        print(f"\n💰 IMPACT ANALYSIS")
        print("-" * 30)
        print(f"Total economic impact: ${impact.get('total_economic_impact_usd', 0):.2f}")
        print(f"Total CO2 emissions: {impact.get('total_co2_emissions_lbs', 0):.1f} lbs")
        print(f"Average delay: {impact.get('average_delay_minutes', 0)} minutes")
    
    # Bottleneck details
    bottlenecks = analysis.get('bottleneck_predictions', [])
    if bottlenecks:
        print(f"\n🚨 BOTTLENECK DETAILS")
        print("-" * 30)
        
        for i, bottleneck in enumerate(bottlenecks, 1):
            print(f"\nBottleneck #{i}: {bottleneck['type'].replace('_', ' ').title()}")
            print(f"  Location: {bottleneck['location']['zone']}")
            print(f"  Probability: {bottleneck['probability']:.2f}")
            print(f"  Severity: {bottleneck['severity']}/5")
            print(f"  Duration: {bottleneck['timing']['estimated_duration_minutes']:.1f} minutes")
            print(f"  Aircraft affected: {len(bottleneck['aircraft_affected'])}")
            
            impact = bottleneck['impact_analysis']
            print(f"  Passengers affected: {impact['passengers_affected']}")
            print(f"  Fuel waste: {impact['fuel_waste_gallons']:.1f} gallons")
            print(f"  Fuel cost: ${impact['fuel_cost_estimate']:.2f}")
            print(f"  CO2 emissions: {impact['co2_emissions_lbs']:.1f} lbs")
            print(f"  Economic impact: ${impact['economic_impact_estimate']:.2f}")
            
            if bottleneck.get('recommended_mitigations'):
                print(f"  Recommended actions:")
                for mitigation in bottleneck['recommended_mitigations'][:2]:
                    print(f"    • {mitigation['action']} (effectiveness: {mitigation['estimated_effectiveness']:.1f})")
    else:
        print("\n✅ No bottlenecks detected - operations running smoothly!")


def main():
    """Main function"""
    print("🚀 Simplified Bottleneck Prediction Demo")
    print("Processing sample aircraft data...")
    print("=" * 50)
    
    # Load sample data
    adsb_data = load_sample_data()
    if not adsb_data:
        return
    
    # Display aircraft information
    print(f"\n📋 Aircraft Information:")
    print("-" * 30)
    for i, aircraft in enumerate(adsb_data['aircraft'], 1):
        flight_id = aircraft.get('flight', 'UNKNOWN')
        aircraft_type = aircraft.get('t', 'UNKNOWN')
        lat = aircraft.get('lat', 0)
        lon = aircraft.get('lon', 0)
        alt = aircraft.get('alt_baro', 0)
        speed = aircraft.get('gs', 0)
        heading = aircraft.get('track', 0)
        
        print(f"{i}. {flight_id} ({aircraft_type})")
        print(f"   Position: {lat:.4f}, {lon:.4f}")
        print(f"   Altitude: {alt} ft, Speed: {speed} kts, Heading: {heading}°")
    
    # Create airport configuration
    airport_config = {
        'icao': 'KJFK',
        'name': 'John F. Kennedy International Airport',
        'city': 'New York',
        'country': 'USA'
    }
    
    # Run prediction
    print(f"\n🔍 Running Bottleneck Analysis...")
    predictor = SimplifiedBottleneckPredictor()
    analysis = predictor.predict_bottlenecks(adsb_data, airport_config)
    
    print("✅ Analysis completed successfully!")
    
    # Display results
    display_results(analysis)
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'simplified_bottleneck_analysis_{timestamp}.json'
    
    with open(filename, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")
    
    print(f"\n🎉 Demo completed successfully!")
    print(f"\n📋 Summary:")
    print(f"- {analysis['total_aircraft_monitored']} aircraft analyzed")
    print(f"- {analysis['airport_summary']['total_bottlenecks_predicted']} bottlenecks predicted")
    print(f"- Overall risk: {analysis['airport_summary']['overall_delay_risk'].upper()}")
    print(f"- Economic impact: ${analysis.get('impact_metrics', {}).get('total_economic_impact_usd', 0):.2f}")


if __name__ == "__main__":
    main()
