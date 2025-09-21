"""
Agentic AI Bottleneck Predictor with Cerebras Integration
Uses Cerebras as an AI agent to analyze bottlenecks comprehensively including:
- Fuel consumption analysis
- Cost impact calculations  
- Passenger impact assessment
- Bottleneck duration predictions
- Aircraft-specific analysis
"""

import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Any
import math
import re
from dataclasses import dataclass, asdict
import asyncio
import aiohttp

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None
    print("Warning: cerebras-cloud-sdk not available. Bottleneck prediction will be disabled.")


@dataclass
class AircraftInfo:
    """Detailed aircraft information for analysis"""
    callsign: str
    aircraft_type: str  # "Boeing 737", "Airbus A320", etc.
    airline: str
    passenger_capacity: int
    fuel_capacity: int  # gallons
    fuel_consumption_rate: float  # gallons per hour
    operating_cost_per_hour: float  # USD
    current_fuel_level: Optional[float] = None
    flight_duration: Optional[int] = None  # minutes


@dataclass
class BottleneckImpact:
    """Comprehensive bottleneck impact analysis"""
    bottleneck_id: str
    location: str
    severity: str  # "Low", "Medium", "High", "Critical"
    estimated_duration: int  # minutes
    aircraft_affected: List[str]
    total_passengers_affected: int
    fuel_waste: float  # gallons
    fuel_cost: float  # USD
    operational_cost: float  # USD
    co2_emissions: float  # lbs
    economic_impact: float  # total USD
    delay_cascade_risk: float  # 0-100%
    mitigation_recommendations: List[str]
    confidence_score: float  # 0-100%


@dataclass
class PilotCommunication:
    """Data structure for pilot communications"""
    timestamp: str
    frequency: str
    pilot_callsign: str
    message: str
    message_type: str
    location_context: Optional[str] = None
    urgency_level: int = 1


class AircraftDatabase:
    """Database of aircraft information for analysis"""
    
    AIRCRAFT_SPECS = {
        "B737": {
            "name": "Boeing 737",
            "passenger_capacity": 180,
            "fuel_capacity": 6875,
            "fuel_consumption_rate": 850,  # gallons/hour
            "operating_cost_per_hour": 2500
        },
        "A320": {
            "name": "Airbus A320", 
            "passenger_capacity": 180,
            "fuel_capacity": 6380,
            "fuel_consumption_rate": 800,
            "operating_cost_per_hour": 2400
        },
        "B777": {
            "name": "Boeing 777",
            "passenger_capacity": 350,
            "fuel_capacity": 18100,
            "fuel_consumption_rate": 2000,
            "operating_cost_per_hour": 4500
        },
        "A350": {
            "name": "Airbus A350",
            "passenger_capacity": 350,
            "fuel_capacity": 17500,
            "fuel_consumption_rate": 1900,
            "operating_cost_per_hour": 4200
        },
        "B787": {
            "name": "Boeing 787",
            "passenger_capacity": 300,
            "fuel_capacity": 12600,
            "fuel_consumption_rate": 1500,
            "operating_cost_per_hour": 3500
        },
        "A330": {
            "name": "Airbus A330",
            "passenger_capacity": 300,
            "fuel_capacity": 13900,
            "fuel_consumption_rate": 1600,
            "operating_cost_per_hour": 3200
        }
    }
    
    AIRLINE_CODES = {
        "DAL": "Delta Air Lines",
        "UAL": "United Airlines", 
        "AAL": "American Airlines",
        "JBU": "JetBlue Airways",
        "SWA": "Southwest Airlines",
        "CPA": "Cathay Pacific",
        "BAW": "British Airways",
        "AFR": "Air France",
        "LUF": "Lufthansa",
        "JAL": "Japan Airlines"
    }
    
    @classmethod
    def get_aircraft_info(cls, callsign: str, aircraft_type: str = None) -> AircraftInfo:
        """Get aircraft information based on callsign and type"""
        # Extract airline code
        airline_code = callsign[:3] if len(callsign) >= 3 else "UNK"
        airline = cls.AIRLINE_CODES.get(airline_code, "Unknown Airline")
        
        # Determine aircraft type if not provided
        if not aircraft_type:
            aircraft_type = cls._infer_aircraft_type(callsign, airline_code)
        
        # Get aircraft specifications
        specs = cls.AIRCRAFT_SPECS.get(aircraft_type, cls.AIRCRAFT_SPECS["B737"])
        
        return AircraftInfo(
            callsign=callsign,
            aircraft_type=specs["name"],
            airline=airline,
            passenger_capacity=specs["passenger_capacity"],
            fuel_capacity=specs["fuel_capacity"],
            fuel_consumption_rate=specs["fuel_consumption_rate"],
            operating_cost_per_hour=specs["operating_cost_per_hour"]
        )
    
    @classmethod
    def _infer_aircraft_type(cls, callsign: str, airline_code: str) -> str:
        """Infer aircraft type based on callsign patterns"""
        # Simple heuristic - in real implementation, use ADS-B data
        if airline_code in ["DAL", "UAL", "AAL"]:
            return "B737"  # Most common for major US carriers
        elif airline_code == "JBU":
            return "A320"  # JetBlue primarily uses A320
        elif airline_code == "SWA":
            return "B737"  # Southwest uses B737
        else:
            return "B737"  # Default


class AgenticCerebrasAnalyzer:
    """AI Agent using Cerebras for comprehensive bottleneck analysis"""
    
    def __init__(self):
        self.cerebras = None
        if Cerebras:
            try:
                import os
                if os.getenv('CEREBRAS_API_KEY'):
                    self.cerebras = Cerebras()
                    print("✅ Cerebras AI Agent initialized successfully")
                else:
                    print("⚠️  CEREBRAS_API_KEY not set - AI agent disabled")
            except Exception as e:
                print(f"⚠️  Cerebras initialization failed: {e} - AI agent disabled")
    
    async def analyze_bottleneck_comprehensively(self, 
                                              aircraft_data: List[Dict], 
                                              communications: List[PilotCommunication],
                                              airport_code: str) -> Dict[str, Any]:
        """Use Cerebras AI agent for comprehensive bottleneck analysis"""
        
        if not self.cerebras:
            return self._fallback_analysis(aircraft_data, communications, airport_code)
        
        try:
            # Prepare comprehensive data for AI analysis
            analysis_data = self._prepare_analysis_data(aircraft_data, communications, airport_code)
            
            # Multi-step AI agent analysis
            ai_analysis = await self._run_agentic_analysis(analysis_data)
            
            return ai_analysis
            
        except Exception as e:
            print(f"Error in AI agent analysis: {e}")
            return self._fallback_analysis(aircraft_data, communications, airport_code)
    
    def _prepare_analysis_data(self, aircraft_data: List[Dict], communications: List[PilotCommunication], airport_code: str) -> Dict:
        """Prepare comprehensive data for AI analysis"""
        
        # Analyze aircraft types and airlines
        aircraft_analysis = []
        total_passengers = 0
        total_fuel_capacity = 0
        
        for ac in aircraft_data:
            callsign = ac.get('flight', 'Unknown')
            aircraft_info = AircraftDatabase.get_aircraft_info(callsign)
            
            aircraft_analysis.append({
                "callsign": callsign,
                "aircraft_type": aircraft_info.aircraft_type,
                "airline": aircraft_info.airline,
                "passenger_capacity": aircraft_info.passenger_capacity,
                "fuel_capacity": aircraft_info.fuel_capacity,
                "fuel_consumption_rate": aircraft_info.fuel_consumption_rate,
                "operating_cost_per_hour": aircraft_info.operating_cost_per_hour,
                "current_position": {
                    "lat": ac.get('lat'),
                    "lon": ac.get('lon'),
                    "altitude": ac.get('altitude'),
                    "speed": ac.get('speed'),
                    "heading": ac.get('heading')
                }
            })
            
            total_passengers += aircraft_info.passenger_capacity
            total_fuel_capacity += aircraft_info.fuel_capacity
        
        # Analyze communications
        comm_analysis = {
            "total_communications": len(communications),
            "frequency_distribution": {},
            "urgency_levels": {},
            "message_types": {},
            "location_contexts": {},
            "recent_messages": []
        }
        
        for comm in communications:
            freq = comm.frequency
            comm_analysis["frequency_distribution"][freq] = comm_analysis["frequency_distribution"].get(freq, 0) + 1
            comm_analysis["urgency_levels"][comm.urgency_level] = comm_analysis["urgency_levels"].get(comm.urgency_level, 0) + 1
            comm_analysis["message_types"][comm.message_type] = comm_analysis["message_types"].get(comm.message_type, 0) + 1
            if comm.location_context:
                comm_analysis["location_contexts"][comm.location_context] = comm_analysis["location_contexts"].get(comm.location_context, 0) + 1
            
            comm_analysis["recent_messages"].append({
                "callsign": comm.pilot_callsign,
                "message": comm.message,
                "type": comm.message_type,
                "urgency": comm.urgency_level,
                "context": comm.location_context
            })
        
        return {
            "airport_code": airport_code,
            "timestamp": datetime.now().isoformat(),
            "aircraft_count": len(aircraft_data),
            "aircraft_analysis": aircraft_analysis,
            "total_passengers": total_passengers,
            "total_fuel_capacity": total_fuel_capacity,
            "communication_analysis": comm_analysis,
            "airport_context": self._get_airport_context(airport_code)
        }
    
    def _get_airport_context(self, airport_code: str) -> Dict:
        """Get airport-specific context for analysis"""
        airport_contexts = {
            "JFK": {
                "name": "John F. Kennedy International Airport",
                "runways": 4,
                "terminals": 6,
                "annual_passengers": 62000000,
                "peak_hour_capacity": 120,
                "typical_delay_factor": 1.2
            },
            "LAX": {
                "name": "Los Angeles International Airport", 
                "runways": 4,
                "terminals": 9,
                "annual_passengers": 88000000,
                "peak_hour_capacity": 150,
                "typical_delay_factor": 1.3
            },
            "PHL": {
                "name": "Philadelphia International Airport",
                "runways": 3,
                "terminals": 7,
                "annual_passengers": 32000000,
                "peak_hour_capacity": 80,
                "typical_delay_factor": 1.1
            }
        }
        
        return airport_contexts.get(airport_code, {
            "name": f"{airport_code} Airport",
            "runways": 2,
            "terminals": 3,
            "annual_passengers": 10000000,
            "peak_hour_capacity": 50,
            "typical_delay_factor": 1.0
        })
    
    async def _run_agentic_analysis(self, analysis_data: Dict) -> Dict[str, Any]:
        """Run multi-step agentic AI analysis using Cerebras"""
        
        # Step 1: Bottleneck Detection and Classification
        bottleneck_analysis = await self._analyze_bottlenecks(analysis_data)
        
        # Step 2: Impact Assessment
        impact_analysis = await self._assess_impacts(analysis_data, bottleneck_analysis)
        
        # Step 3: Cost Analysis
        cost_analysis = await self._analyze_costs(analysis_data, bottleneck_analysis)
        
        # Step 4: Mitigation Strategies
        mitigation_analysis = await self._generate_mitigation_strategies(analysis_data, bottleneck_analysis, impact_analysis)
        
        # Step 5: Risk Assessment
        risk_analysis = await self._assess_risks(analysis_data, bottleneck_analysis, impact_analysis)
        
        return {
            "analysis_type": "agentic_ai_comprehensive",
            "timestamp": analysis_data["timestamp"],
            "airport": analysis_data["airport_code"],
            "bottleneck_analysis": bottleneck_analysis,
            "impact_analysis": impact_analysis,
            "cost_analysis": cost_analysis,
            "mitigation_analysis": mitigation_analysis,
            "risk_analysis": risk_analysis,
            "confidence_score": self._calculate_overall_confidence(analysis_data),
            "ai_agent_version": "1.0.0"
        }
    
    async def _analyze_bottlenecks(self, data: Dict) -> Dict:
        """AI analysis of bottleneck patterns and locations"""
        
        prompt = f"""
        As an AI aviation expert, analyze the following airport traffic data using EXPLICIT RULESETS:

        EXPLICIT BOTTLENECK DETECTION RULESET:
        
        1. GROUND AIRCRAFT ANALYSIS:
           - Count aircraft with altitude='ground' and speed=0
           - RULE: >20 ground aircraft = CRITICAL bottleneck
           - RULE: 15-20 ground aircraft = HIGH bottleneck  
           - RULE: 10-15 ground aircraft = MEDIUM bottleneck
           - RULE: 5-10 ground aircraft = LOW bottleneck
           - RULE: <5 ground aircraft = MINIMAL bottleneck
        
        2. GEOGRAPHIC CLUSTERING:
           - RULE: Aircraft within 0.01 degrees (~1km) = congestion hotspot
           - RULE: >5 aircraft in cluster = significant bottleneck
           - RULE: 3-5 aircraft in cluster = moderate bottleneck
           - RULE: <3 aircraft in cluster = minor bottleneck
        
        3. SPEED-BASED ANALYSIS:
           - RULE: Speed=0 = stationary bottleneck (gate/taxiway)
           - RULE: Speed<50 = slow taxiing bottleneck
           - RULE: Speed>100 = normal operations
        
        4. DENSITY CALCULATION:
           - RULE: >25 aircraft/km² = CRITICAL density
           - RULE: 20-25 aircraft/km² = HIGH density
           - RULE: 15-20 aircraft/km² = MEDIUM density
           - RULE: 10-15 aircraft/km² = LOW density
           - RULE: <10 aircraft/km² = MINIMAL density
        
        5. SEVERITY SCORING RULES:
           - CRITICAL (90-100): >25 aircraft OR multiple hotspots OR emergency
           - HIGH (70-89): 20-25 aircraft OR 3+ hotspots OR significant delays
           - MEDIUM (50-69): 15-20 aircraft OR 2-3 hotspots OR moderate delays
           - LOW (30-49): 10-15 aircraft OR 1-2 hotspots OR minor delays
           - MINIMAL (0-29): <10 aircraft OR no clear hotspots
        
        THOUGHT PROCESS:
        1. Count and categorize aircraft by altitude/speed
        2. Identify geographic clusters using lat/lon proximity
        3. Calculate density metrics and congestion scores
        4. Apply severity rules based on aircraft count and distribution
        5. Generate specific bottleneck locations and recommendations
        
        CURRENT DATA:
        Airport: {data['airport_code']} ({data['airport_context']['name']})
        Aircraft Count: {data['aircraft_count']}
        Total Passengers: {data['total_passengers']}
        
        Aircraft Analysis:
        {json.dumps(data['aircraft_analysis'][:10], indent=2)}
        
        Communication Analysis:
        - Total Communications: {data['communication_analysis']['total_communications']}
        - Frequency Distribution: {data['communication_analysis']['frequency_distribution']}
        - Urgency Levels: {data['communication_analysis']['urgency_levels']}
        - Message Types: {data['communication_analysis']['message_types']}
        - Location Contexts: {data['communication_analysis']['location_contexts']}
        
        Recent Communications:
        {json.dumps(data['communication_analysis']['recent_messages'][:5], indent=2)}
        
        Airport Context:
        - Runways: {data['airport_context']['runways']}
        - Terminals: {data['airport_context']['terminals']}
        - Peak Hour Capacity: {data['airport_context']['peak_hour_capacity']}
        - Typical Delay Factor: {data['airport_context']['typical_delay_factor']}
        
        REQUIRED OUTPUT FORMAT (JSON):
        {{
            "bottleneck_locations": [
                {{
                    "location": "Gate A1-A5",
                    "severity": "High",
                    "aircraft_count": 8,
                    "estimated_duration": "45 minutes",
                    "contributing_factors": ["Multiple pushback requests", "Ground crew shortage"]
                }}
            ],
            "overall_severity": "High",
            "severity_score": 75,
            "total_bottlenecks": 3,
            "estimated_resolution_time": "60 minutes",
            "analysis_summary": "Detailed explanation of findings"
        }}
        
        Apply the ruleset above and provide analysis in the exact JSON format specified.
        """
        
        response = self.cerebras.completions.create(
            model="llama-3.3-70b",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.2
        )
        
        return {
            "ai_analysis": response.choices[0].text if hasattr(response, 'choices') else str(response),
            "bottleneck_count": len(data['aircraft_analysis']),
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _assess_impacts(self, data: Dict, bottleneck_analysis: Dict) -> Dict:
        """AI assessment of bottleneck impacts"""
        
        prompt = f"""
        As an AI aviation impact analyst, assess comprehensive impacts using EXPLICIT RULESETS:

        EXPLICIT IMPACT ASSESSMENT RULESET:
        
        1. PASSENGER IMPACT CALCULATION:
           - RULE: Average delay = bottleneck_duration × delay_multiplier
           - RULE: delay_multiplier = 1.2 for ground bottlenecks, 1.5 for runway bottlenecks
           - RULE: Passenger satisfaction = 100 - (delay_minutes × 0.5)
           - RULE: Connection impact = 30% of delayed passengers miss connections
        
        2. FUEL IMPACT CALCULATION:
           - RULE: Ground aircraft fuel burn = 200-400 lbs/hour (varies by aircraft type)
           - RULE: Taxiing aircraft fuel burn = 100-200 lbs/hour
           - RULE: Fuel cost = $3.50/gallon average
           - RULE: CO2 emissions = fuel_burned × 3.15 lbs CO2/lb fuel
        
        3. OPERATIONAL IMPACT RULES:
           - RULE: Aircraft utilization loss = delay_hours × $500/hour
           - RULE: Crew overtime = delay >2 hours = $200/hour overtime
           - RULE: Gate utilization = 85% efficiency during bottlenecks
           - RULE: Maintenance impact = delay >4 hours = additional inspection required
        
        4. ECONOMIC IMPACT RULES:
           - RULE: Direct costs = fuel_waste + crew_overtime + gate_fees
           - RULE: Passenger compensation = $200 per passenger for delays >3 hours
           - RULE: Airline revenue loss = delayed_flights × average_ticket_price × 0.1
           - RULE: Airport revenue impact = delayed_operations × $50 per operation
        
        5. SAFETY IMPACT RULES:
           - RULE: Safety risk = HIGH if >20 aircraft on ground simultaneously
           - RULE: Communication stress = HIGH if >50 communications/hour
           - RULE: Emergency response = DELAYED if >15 aircraft in emergency area
        
        THOUGHT PROCESS:
        1. Extract bottleneck duration and severity from analysis
        2. Apply aircraft-specific fuel burn rates
        3. Calculate passenger delay impacts using rules
        4. Compute economic costs using industry standards
        5. Assess safety implications based on traffic density
        
        CURRENT DATA:
        Bottleneck Analysis: {bottleneck_analysis['ai_analysis']}
        
        Aircraft Fleet Analysis:
        {json.dumps(data['aircraft_analysis'], indent=2)}
        
        Total Passengers Affected: {data['total_passengers']}
        Total Fuel Capacity: {data['total_fuel_capacity']} gallons
        
        REQUIRED OUTPUT FORMAT (JSON):
        {{
            "passenger_impact": {{
                "total_passengers_affected": 4320,
                "average_delay_minutes": 45,
                "passenger_satisfaction_score": 77.5,
                "missed_connections": 1296
            }},
            "fuel_impact": {{
                "total_fuel_waste_gallons": 2500,
                "fuel_cost_usd": 8750,
                "co2_emissions_lbs": 7875
            }},
            "operational_impact": {{
                "aircraft_utilization_loss_usd": 15000,
                "crew_overtime_usd": 4000,
                "gate_efficiency_percent": 85
            }},
            "economic_impact": {{
                "direct_costs_usd": 25000,
                "passenger_compensation_usd": 864000,
                "airline_revenue_loss_usd": 50000,
                "total_economic_impact_usd": 939000
            }},
            "safety_assessment": {{
                "safety_risk_level": "Medium",
                "communication_stress_level": "High",
                "emergency_response_status": "Normal"
            }}
        }}
        
        Apply the ruleset above and provide analysis in the exact JSON format specified.
        """
        
        response = self.cerebras.completions.create(
            model="llama-3.3-70b",
            prompt=prompt,
            max_tokens=1200,
            temperature=0.2
        )
        
        return {
            "impact_assessment": response.choices[0].text if hasattr(response, 'choices') else str(response),
            "passengers_affected": data['total_passengers'],
            "fuel_capacity_at_risk": data['total_fuel_capacity'],
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    async def _analyze_costs(self, data: Dict, bottleneck_analysis: Dict) -> Dict:
        """AI analysis of financial costs and economic impact"""
        
        prompt = f"""
        As an AI aviation cost analyst, calculate comprehensive costs using EXPLICIT RULESETS:

        EXPLICIT COST ANALYSIS RULESET:
        
        1. DIRECT OPERATIONAL COSTS:
           - RULE: Fuel cost = fuel_waste_gallons × $3.50/gallon
           - RULE: Crew overtime = delay_hours × $200/hour × crew_size
           - RULE: Aircraft operating cost = delay_hours × $500/hour × aircraft_count
           - RULE: Ground handling = $150 per delayed aircraft
        
        2. PASSENGER COMPENSATION COSTS:
           - RULE: EU261 compensation = €600 for delays >4 hours
           - RULE: US compensation = $200 for delays >3 hours
           - RULE: Meal vouchers = $15 per passenger for delays >2 hours
           - RULE: Hotel accommodation = $150/night for overnight delays
        
        3. AIRLINE REVENUE IMPACT:
           - RULE: Revenue loss = delayed_flights × average_ticket_price × 0.1
           - RULE: Customer retention cost = 5% of passengers switch airlines
           - RULE: Reputation damage = $10,000 per major delay incident
           - RULE: Future booking impact = 2% reduction for 3 months
        
        4. AIRPORT REVENUE IMPACT:
           - RULE: Landing fees = $50 per delayed operation
           - RULE: Terminal fees = $25 per delayed passenger
           - RULE: Parking fees = $100/hour for delayed aircraft
           - RULE: Ground services = $200 per delayed aircraft
        
        5. ENVIRONMENTAL COSTS:
           - RULE: Carbon tax = $50 per ton CO2 emissions
           - RULE: Noise pollution = $10 per affected resident
           - RULE: Air quality impact = $25 per ton NOx emissions
           - RULE: Environmental fines = $5,000 per violation
        
        THOUGHT PROCESS:
        1. Extract delay duration and aircraft count from bottleneck analysis
        2. Apply industry-standard cost rates for each category
        3. Calculate passenger compensation based on delay duration
        4. Compute revenue impacts using historical data
        5. Assess environmental costs using emission factors
        
        CURRENT DATA:
        Bottleneck Analysis: {bottleneck_analysis['ai_analysis']}
        
        Aircraft Cost Data:
        {json.dumps([{
            "callsign": ac["callsign"],
            "aircraft_type": ac["aircraft_type"],
            "airline": ac["airline"],
            "fuel_consumption_rate": ac["fuel_consumption_rate"],
            "operating_cost_per_hour": ac["operating_cost_per_hour"],
            "passenger_capacity": ac["passenger_capacity"]
        } for ac in data['aircraft_analysis']], indent=2)}
        
        REQUIRED OUTPUT FORMAT (JSON):
        {{
            "direct_costs": {{
                "fuel_cost_usd": 8750,
                "crew_overtime_usd": 4000,
                "aircraft_operating_cost_usd": 15000,
                "ground_handling_usd": 3000,
                "total_direct_costs_usd": 30750
            }},
            "passenger_compensation": {{
                "delay_compensation_usd": 864000,
                "meal_vouchers_usd": 64800,
                "hotel_accommodation_usd": 0,
                "total_passenger_costs_usd": 928800
            }},
            "revenue_impact": {{
                "airline_revenue_loss_usd": 50000,
                "customer_retention_cost_usd": 25000,
                "reputation_damage_usd": 10000,
                "future_booking_impact_usd": 15000,
                "total_revenue_impact_usd": 100000
            }},
            "airport_costs": {{
                "landing_fees_usd": 1500,
                "terminal_fees_usd": 108000,
                "parking_fees_usd": 2000,
                "ground_services_usd": 6000,
                "total_airport_costs_usd": 117500
            }},
            "environmental_costs": {{
                "carbon_tax_usd": 394,
                "noise_pollution_usd": 5000,
                "air_quality_usd": 200,
                "environmental_fines_usd": 0,
                "total_environmental_costs_usd": 5594
            }},
            "total_cost_summary": {{
                "total_economic_impact_usd": 1158644,
                "cost_per_passenger_usd": 268,
                "cost_per_aircraft_usd": 48277
            }}
        }}
        
        Apply the ruleset above and provide analysis in the exact JSON format specified.
        """
        
        response = self.cerebras.completions.create(
            model="llama-3.3-70b",
            prompt=prompt,
            max_tokens=1200,
            temperature=0.2
        )
        
        return {
            "cost_analysis": response.choices[0].text if hasattr(response, 'choices') else str(response),
            "aircraft_count": data['aircraft_count'],
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    async def _generate_mitigation_strategies(self, data: Dict, bottleneck_analysis: Dict, impact_analysis: Dict) -> Dict:
        """AI generation of mitigation strategies"""
        
        prompt = f"""
        As an AI aviation operations expert, generate comprehensive mitigation strategies:

        Bottleneck Analysis: {bottleneck_analysis['ai_analysis']}
        Impact Analysis: {impact_analysis['impact_assessment']}
        
        Current Situation:
        - Airport: {data['airport_code']}
        - Aircraft Count: {data['aircraft_count']}
        - Passengers Affected: {data['total_passengers']}
        - Communication Patterns: {data['communication_analysis']['message_types']}
        
        Generate mitigation strategies for:
        
        1. Immediate Actions (0-30 minutes):
           - Ground stop procedures
           - Runway/taxiway optimization
           - Communication protocols
           - Emergency procedures
        
        2. Short-term Actions (30 minutes - 2 hours):
           - Traffic flow management
           - Aircraft rerouting
           - Gate reassignment
           - Crew resource management
        
        3. Medium-term Actions (2-6 hours):
           - Schedule adjustments
           - Aircraft swaps
           - Maintenance coordination
           - Passenger management
        
        4. Long-term Actions (6+ hours):
           - Operational procedure changes
           - Infrastructure improvements
           - Policy adjustments
           - Technology enhancements
        
        5. Communication Strategies:
           - Pilot communication protocols
           - ATC coordination
           - Passenger communication
           - Stakeholder notification
        
        6. Technology Solutions:
           - AI/ML optimization
           - Real-time monitoring
           - Predictive analytics
           - Automated decision support
        
        Provide specific, actionable strategies with implementation timelines and expected outcomes.
        """
        
        response = self.cerebras.completions.create(
            model="llama-3.3-70b",
            prompt=prompt,
            max_tokens=1200,
            temperature=0.3
        )
        
        return {
            "mitigation_strategies": response.choices[0].text if hasattr(response, 'choices') else str(response),
            "strategy_count": 6,
            "generation_timestamp": datetime.now().isoformat()
        }
    
    async def _assess_risks(self, data: Dict, bottleneck_analysis: Dict, impact_analysis: Dict) -> Dict:
        """AI risk assessment and cascade analysis"""
        
        prompt = f"""
        As an AI aviation risk analyst, assess comprehensive risks and cascade effects:

        Bottleneck Analysis: {bottleneck_analysis['ai_analysis']}
        Impact Analysis: {impact_analysis['impact_assessment']}
        
        Current Risk Factors:
        - Aircraft Count: {data['aircraft_count']}
        - Communication Volume: {data['communication_analysis']['total_communications']}
        - Urgency Levels: {data['communication_analysis']['urgency_levels']}
        - Airport Capacity: {data['airport_context']['peak_hour_capacity']}
        
        Assess risks in these categories:
        
        1. Operational Risks:
           - Safety risks
           - Communication failures
           - Equipment failures
           - Human factor risks
        
        2. Cascade Risks:
           - Delay propagation
           - Network-wide impacts
           - Hub airport effects
           - International connection impacts
        
        3. Weather Risks:
           - Current weather impact
           - Forecasted weather risks
           - Seasonal factors
           - Extreme weather scenarios
        
        4. Economic Risks:
           - Financial losses
           - Market impact
           - Competitive disadvantage
           - Regulatory penalties
        
        5. Reputation Risks:
           - Customer satisfaction
           - Brand damage
           - Media attention
           - Regulatory scrutiny
        
        6. System Risks:
           - Infrastructure stress
           - Technology failures
           - Staff fatigue
           - Resource depletion
        
        7. Mitigation Risks:
           - Strategy effectiveness
           - Implementation challenges
           - Unintended consequences
           - Resource constraints
        
        Provide risk assessment with probability scores, impact levels, and risk mitigation priorities.
        """
        
        response = self.cerebras.completions.create(
            model="llama-3.3-70b",
            prompt=prompt,
            max_tokens=1200,
            temperature=0.2
        )
        
        return {
            "risk_assessment": response.choices[0].text if hasattr(response, 'choices') else str(response),
            "risk_categories": 7,
            "assessment_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_overall_confidence(self, data: Dict) -> float:
        """Calculate overall confidence score for the analysis"""
        confidence = 70  # Base confidence
        
        # Data quality factors
        if data['aircraft_count'] > 10:
            confidence += 10
        if data['communication_analysis']['total_communications'] > 5:
            confidence += 10
        if len(data['aircraft_analysis']) > 5:
            confidence += 5
        
        # Communication coverage
        comm_coverage = data['communication_analysis']['total_communications'] / max(data['aircraft_count'], 1)
        if comm_coverage > 0.5:
            confidence += 5
        
        return min(100, confidence)
    
    def _fallback_analysis(self, aircraft_data: List[Dict], communications: List[PilotCommunication], airport_code: str) -> Dict:
        """Fallback analysis when Cerebras is not available"""
        
        # Simple heuristic analysis
        aircraft_count = len(aircraft_data)
        comm_count = len(communications)
        
        # Basic bottleneck detection
        bottleneck_severity = "Low"
        if aircraft_count > 20:
            bottleneck_severity = "High"
        elif aircraft_count > 15:
            bottleneck_severity = "Medium"
        
        # Basic impact calculation
        total_passengers = sum(AircraftDatabase.get_aircraft_info(ac.get('flight', 'Unknown')).passenger_capacity for ac in aircraft_data)
        estimated_delay = 30 if bottleneck_severity == "High" else 15 if bottleneck_severity == "Medium" else 5
        
        return {
            "analysis_type": "fallback_heuristic",
            "timestamp": datetime.now().isoformat(),
            "airport": airport_code,
            "bottleneck_analysis": {
                "severity": bottleneck_severity,
                "aircraft_count": aircraft_count,
                "estimated_delay_minutes": estimated_delay
            },
            "impact_analysis": {
                "passengers_affected": total_passengers,
                "estimated_delay_per_passenger": estimated_delay
            },
            "cost_analysis": {
                "estimated_cost_per_hour": aircraft_count * 2000,
                "total_estimated_cost": aircraft_count * 2000 * (estimated_delay / 60)
            },
            "mitigation_analysis": {
                "recommendations": [
                    "Monitor traffic density",
                    "Consider spacing adjustments",
                    "Prepare contingency plans"
                ]
            },
            "risk_analysis": {
                "risk_level": bottleneck_severity,
                "cascade_risk": "Low" if aircraft_count < 15 else "Medium"
            },
            "confidence_score": 60.0,
            "ai_agent_version": "fallback"
        }


class AgenticBottleneckPredictor:
    """Main predictor class with agentic AI capabilities"""
    
    def __init__(self):
        # Initialize Cerebras AI agent
        self.ai_analyzer = AgenticCerebrasAnalyzer()
        self.results_file = "model/results.txt"
        
        # Initialize aircraft database
        self.aircraft_db = AircraftDatabase()
        
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
    
    def simulate_pilot_communications(self, aircraft_data: List[Dict]) -> List[PilotCommunication]:
        """Simulate pilot communications based on aircraft data"""
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
    
    async def predict_and_analyze(self, aircraft_data: List[Dict], airport_code: str, communications: List[PilotCommunication] = None) -> Dict:
        """Main method for comprehensive AI-powered bottleneck analysis"""
        
        # Simulate communications if not provided
        if not communications:
            communications = self.simulate_pilot_communications(aircraft_data)
        
        # Run comprehensive AI analysis
        ai_analysis = await self.ai_analyzer.analyze_bottleneck_comprehensively(
            aircraft_data, communications, airport_code
        )
        
        # Add traffic analysis
        traffic_analysis = self.analyze_traffic_density(aircraft_data)
        ai_analysis["traffic_analysis"] = traffic_analysis
        
        return ai_analysis
    
    def save_results(self, analysis_results: Dict):
        """Save comprehensive analysis results to results.txt"""
        try:
            with open(self.results_file, 'a') as f:
                f.write(f"\n=== Agentic AI Bottleneck Analysis - {analysis_results['timestamp']} ===\n")
                f.write(f"Airport: {analysis_results['airport']}\n")
                f.write(f"Analysis Type: {analysis_results['analysis_type']}\n")
                f.write(f"AI Agent Version: {analysis_results.get('ai_agent_version', 'Unknown')}\n")
                f.write(f"Confidence Score: {analysis_results.get('confidence_score', 0):.1f}%\n")
                
                # Traffic Analysis
                if 'traffic_analysis' in analysis_results:
                    traffic = analysis_results['traffic_analysis']
                    f.write(f"\nTraffic Analysis:\n")
                    f.write(f"  - Total Aircraft: {traffic['ground_aircraft'] + traffic['low_alt_aircraft'] + traffic['high_alt_aircraft']}\n")
                    f.write(f"  - Ground: {traffic['ground_aircraft']}\n")
                    f.write(f"  - Low Altitude: {traffic['low_alt_aircraft']}\n")
                    f.write(f"  - High Altitude: {traffic['high_alt_aircraft']}\n")
                    f.write(f"  - Density Score: {traffic['density_score']:.1f}/100\n")
                    f.write(f"  - Hotspots: {len(traffic['hotspots'])}\n")
                
                # Bottleneck Analysis
                if 'bottleneck_analysis' in analysis_results:
                    bottleneck = analysis_results['bottleneck_analysis']
                    f.write(f"\nBottleneck Analysis:\n")
                    f.write(f"  - AI Analysis: {bottleneck.get('ai_analysis', 'N/A')[:200]}...\n")
                    f.write(f"  - Bottleneck Count: {bottleneck.get('bottleneck_count', 0)}\n")
                
                # Impact Analysis
                if 'impact_analysis' in analysis_results:
                    impact = analysis_results['impact_analysis']
                    f.write(f"\nImpact Analysis:\n")
                    f.write(f"  - Passengers Affected: {impact.get('passengers_affected', 0)}\n")
                    f.write(f"  - Fuel Capacity at Risk: {impact.get('fuel_capacity_at_risk', 0)} gallons\n")
                    f.write(f"  - AI Assessment: {impact.get('impact_assessment', 'N/A')[:200]}...\n")
                
                # Cost Analysis
                if 'cost_analysis' in analysis_results:
                    cost = analysis_results['cost_analysis']
                    f.write(f"\nCost Analysis:\n")
                    f.write(f"  - Aircraft Count: {cost.get('aircraft_count', 0)}\n")
                    f.write(f"  - AI Cost Analysis: {cost.get('cost_analysis', 'N/A')[:200]}...\n")
                
                # Mitigation Analysis
                if 'mitigation_analysis' in analysis_results:
                    mitigation = analysis_results['mitigation_analysis']
                    f.write(f"\nMitigation Strategies:\n")
                    f.write(f"  - Strategy Count: {mitigation.get('strategy_count', 0)}\n")
                    f.write(f"  - AI Strategies: {mitigation.get('mitigation_strategies', 'N/A')[:200]}...\n")
                
                # Risk Analysis
                if 'risk_analysis' in analysis_results:
                    risk = analysis_results['risk_analysis']
                    f.write(f"\nRisk Assessment:\n")
                    f.write(f"  - Risk Categories: {risk.get('risk_categories', 0)}\n")
                    f.write(f"  - AI Risk Assessment: {risk.get('risk_assessment', 'N/A')[:200]}...\n")
                
                f.write("\n" + "="*80 + "\n")
                
        except Exception as e:
            print(f"Error saving results: {e}")


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
    
    async def main():
        predictor = AgenticBottleneckPredictor()
        
        # Run comprehensive analysis
        results = await predictor.predict_and_analyze(sample_aircraft, "JFK")
        
        # Save results
        predictor.save_results(results)
        print("Agentic AI analysis completed and saved to results.txt")
    
    # Run the async main function
    asyncio.run(main())
