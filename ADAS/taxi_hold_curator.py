#!/usr/bin/env python3
"""
Taxi & Hold Scenario Curator - Specialized Tool for Ground Movement and Hold Scenarios

This script creates curated taxi and hold scenarios specifically for training
the ADAS TaskAgent to handle ground traffic management, taxi conflicts, and
hold short situations.

Usage:
    python3.9 taxi_hold_curator.py
    python3.9 taxi_hold_curator.py --scenarios 50 --output-file taxi_hold_scenarios.json
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List
import random
import uuid

from cleaning_agent import CleaningAgent
from config import get_config


class TaxiHoldScenarioCurator:
    """Creates curated taxi and hold scenarios for ADAS training"""

    def __init__(self, cerebras_api_key: str, base_path: str):
        self.cleaning_agent = CleaningAgent(cerebras_api_key)
        self.base_path = base_path
        self.scenarios_dir = os.path.join(base_path, "taxi-hold-scenarios")
        os.makedirs(self.scenarios_dir, exist_ok=True)

    async def generate_taxi_hold_scenarios(
        self, num_scenarios: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate taxi and hold scenarios for training"""

        scenarios = []
        scenario_types = [
            "taxi_conflict_intersection",
            "hold_short_runway",
            "taxi_sequence_optimization",
            "hold_for_departure",
            "taxi_gate_to_runway",
            "hold_for_crossing_traffic",
            "taxi_priority_emergency",
            "hold_weather_delay",
            "taxi_congestion_alley",
            "hold_maintenance_blocking",
        ]

        print(f"🚕 Generating {num_scenarios} taxi and hold scenarios...")

        for i in range(num_scenarios):
            scenario_type = random.choice(scenario_types)
            scenario = await self._create_scenario(scenario_type, i + 1)
            scenarios.append(scenario)

            if (i + 1) % 10 == 0:
                print(f"   Generated {i + 1}/{num_scenarios} scenarios...")

        return scenarios

    async def _create_scenario(
        self, scenario_type: str, scenario_num: int
    ) -> Dict[str, Any]:
        """Create a specific type of taxi/hold scenario"""

        base_scenario = {
            "record_id": str(uuid.uuid4()),
            "scenario_type": scenario_type,
            "scenario_number": scenario_num,
            "timestamp": datetime.now().isoformat(),
            "airport": "KJFK",
        }

        if scenario_type == "taxi_conflict_intersection":
            return await self._create_taxi_intersection_conflict(base_scenario)
        elif scenario_type == "hold_short_runway":
            return await self._create_hold_short_runway(base_scenario)
        elif scenario_type == "taxi_sequence_optimization":
            return await self._create_taxi_sequence_optimization(base_scenario)
        elif scenario_type == "hold_for_departure":
            return await self._create_hold_for_departure(base_scenario)
        elif scenario_type == "taxi_gate_to_runway":
            return await self._create_taxi_gate_to_runway(base_scenario)
        elif scenario_type == "hold_for_crossing_traffic":
            return await self._create_hold_for_crossing_traffic(base_scenario)
        elif scenario_type == "taxi_priority_emergency":
            return await self._create_taxi_priority_emergency(base_scenario)
        elif scenario_type == "hold_weather_delay":
            return await self._create_hold_weather_delay(base_scenario)
        elif scenario_type == "taxi_congestion_alley":
            return await self._create_taxi_congestion_alley(base_scenario)
        elif scenario_type == "hold_maintenance_blocking":
            return await self._create_hold_maintenance_blocking(base_scenario)
        else:
            return await self._create_generic_taxi_hold_scenario(base_scenario)

    async def _create_taxi_intersection_conflict(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a taxi intersection conflict scenario"""

        aircraft_states = {"all_aircraft": []}

        # Aircraft 1: Approaching intersection from east
        callsign1 = f"{random.choice(['AAL', 'DAL'])}{random.randint(100, 999)}"
        aircraft1 = {
            "callsign": callsign1,
            "aircraft_type": "B738",
            "lat": 40.6420,
            "lon": -73.7785,
            "altitude": "ground",
            "speed": 12.0,
            "heading": 270,  # Westbound
            "flight_phase": "taxi_to_runway",
            "runway_proximity": "taxiway_A",
            "airport_area": "east_field",
            "destination_runway": "22L",
            "taxi_clearance": "Taxiway A to runway 22L",
            "distance_to_intersection": 200,  # meters
        }

        # Aircraft 2: Approaching same intersection from south
        callsign2 = f"{random.choice(['JBU', 'UAL'])}{random.randint(100, 999)}"
        aircraft2 = {
            "callsign": callsign2,
            "aircraft_type": "A320",
            "lat": 40.6415,
            "lon": -73.7790,
            "altitude": "ground",
            "speed": 15.0,
            "heading": 360,  # Northbound
            "flight_phase": "taxi_to_gate",
            "runway_proximity": "taxiway_B",
            "airport_area": "south_field",
            "destination_gate": "A12",
            "taxi_clearance": "Taxiway B to gate A12",
            "distance_to_intersection": 150,  # meters
        }

        aircraft_states["all_aircraft"] = [aircraft1, aircraft2]

        expected_action = {
            "command_type": "taxi_control",
            "details": f"{callsign2} hold short of Taxiway A, give way to {callsign1}",
            "affected_aircraft": [callsign2],
            "priority": "high",
            "reasoning": f"{callsign1} has right of way for runway access, {callsign2} should hold",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "taxi_intersection_conflict",
                "severity": "high",
                "intersection": "Taxiway A/B",
                "conflict_distance": "50 meters",
                "estimated_delay": "2 minutes",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Runway-bound traffic typically has priority over gate-bound traffic",
        }

    async def _create_hold_short_runway(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Create a hold short runway scenario"""

        aircraft_states = {"all_aircraft": []}

        # Aircraft ready for departure but runway occupied
        departure_callsign = f"{random.choice(['AAL', 'DAL'])}{random.randint(100, 999)}"
        departure_aircraft = {
            "callsign": departure_callsign,
            "aircraft_type": "B757",
            "lat": 40.6413,
            "lon": -73.7785,
            "altitude": "ground",
            "speed": 0.0,
            "heading": 220,
            "flight_phase": "ready_for_departure",
            "runway_proximity": "runway_22L_threshold",
            "airport_area": "runway_approach",
            "destination": "KLAX",
            "clearance_status": "cleared_for_takeoff",
        }

        # Landing aircraft on final approach
        landing_callsign = f"{random.choice(['JBU', 'UAL'])}{random.randint(100, 999)}"
        landing_aircraft = {
            "callsign": landing_callsign,
            "aircraft_type": "A321",
            "lat": 40.6313,
            "lon": -73.7881,
            "altitude": 500,
            "speed": 140,
            "heading": 220,
            "flight_phase": "final_approach",
            "runway_proximity": "approach_22L",
            "airport_area": "approach_corridor",
            "distance_to_runway": 2.5,  # miles
            "origin": "KBOS",
        }

        aircraft_states["all_aircraft"] = [departure_aircraft, landing_aircraft]

        expected_action = {
            "command_type": "runway_control",
            "details": f"{departure_callsign} hold short runway 22L, traffic on final",
            "affected_aircraft": [departure_callsign],
            "priority": "urgent",
            "reasoning": "Landing traffic has priority, departure must hold until runway is clear",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "hold_short_runway",
                "severity": "urgent",
                "runway": "22L",
                "landing_traffic_distance": "2.5 miles",
                "estimated_hold_time": "3 minutes",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Landing aircraft always have priority over departing aircraft",
        }

    async def _create_taxi_sequence_optimization(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a taxi sequence optimization scenario"""

        aircraft_states = {"all_aircraft": []}

        # Multiple aircraft taxiing to same runway in suboptimal order
        aircraft_data = [
            ("AAL123", "B738", "A15", 5, "KMIA"),  # Close gate, long flight
            ("DAL456", "A320", "B22", 8, "KBOS"),  # Far gate, short flight
            ("JBU789", "B757", "A18", 6, "KLAX"),  # Medium gate, long flight
        ]

        for i, (callsign, aircraft_type, gate, taxi_time, destination) in enumerate(
            aircraft_data
        ):
            aircraft = {
                "callsign": callsign,
                "aircraft_type": aircraft_type,
                "lat": 40.6400 + (i * 0.001),
                "lon": -73.7800 + (i * 0.001),
                "altitude": "ground",
                "speed": 8.0 + (i * 2),
                "heading": 180,
                "flight_phase": "taxi_to_runway",
                "runway_proximity": f"taxiway_sequence_{i+1}",
                "airport_area": "terminal_area",
                "departure_gate": gate,
                "destination": destination,
                "estimated_taxi_time": taxi_time,
                "departure_slot": f"0{8+i}:15",
            }
            aircraft_states["all_aircraft"].append(aircraft)

        # Optimize sequence: shortest taxi time first for efficiency
        expected_action = {
            "command_type": "taxi_sequencing",
            "details": f"AAL123 taxi via Alpha to runway 22L, expedite for slot time",
            "affected_aircraft": ["AAL123"],
            "priority": "medium",
            "reasoning": "AAL123 has shortest taxi time and can make departure slot efficiently",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "taxi_sequence_optimization",
                "severity": "medium",
                "runway": "22L",
                "aircraft_count": 3,
                "optimization_potential": "2 minutes saved",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Optimize taxi sequence based on departure slots and taxi times",
        }

    async def _create_hold_for_departure(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Create a hold for departure scenario"""

        aircraft_states = {"all_aircraft": []}

        # Aircraft holding for departure clearance
        callsign = f"{random.choice(['AAL', 'DAL', 'UAL'])}{random.randint(100, 999)}"
        aircraft = {
            "callsign": callsign,
            "aircraft_type": "B787",
            "lat": 40.6413,
            "lon": -73.7781,
            "altitude": "ground",
            "speed": 0.0,
            "heading": 220,
            "flight_phase": "holding_for_departure",
            "runway_proximity": "runway_22L_holding_point",
            "airport_area": "departure_area",
            "destination": "EGLL",  # London Heathrow
            "departure_slot": "08:45",
            "current_time": "08:42",
            "hold_reason": "departure_spacing",
            "fuel_remaining_minutes": 480,  # 8 hours
        }

        aircraft_states["all_aircraft"] = [aircraft]

        expected_action = {
            "command_type": "departure_hold",
            "details": f"{callsign} continue holding, departure clearance in 3 minutes for slot time",
            "affected_aircraft": [callsign],
            "priority": "medium",
            "reasoning": "Aircraft must wait for assigned departure slot to maintain flow control",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "hold_for_departure_slot",
                "severity": "low",
                "hold_reason": "slot_timing",
                "estimated_hold_time": "3 minutes",
                "departure_slot": "08:45",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Departure slots must be maintained for flow control and spacing",
        }

    async def _create_taxi_gate_to_runway(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Create a taxi from gate to runway scenario"""

        aircraft_states = {"all_aircraft": []}

        callsign = f"{random.choice(['JBU', 'SWA'])}{random.randint(100, 999)}"
        aircraft = {
            "callsign": callsign,
            "aircraft_type": "B737",
            "lat": 40.6395,
            "lon": -73.7805,
            "altitude": "ground",
            "speed": 0.0,
            "heading": 90,
            "flight_phase": "pushback_complete",
            "runway_proximity": "gate_B15",
            "airport_area": "terminal_B",
            "departure_gate": "B15",
            "destination": "KORD",
            "assigned_runway": "22R",
            "taxi_route": "B, A, A7 to runway 22R",
            "passengers": 142,
        }

        aircraft_states["all_aircraft"] = [aircraft]

        expected_action = {
            "command_type": "taxi_clearance",
            "details": f"{callsign} taxi via Bravo, Alpha, Alpha-7 to runway 22R",
            "affected_aircraft": [callsign],
            "priority": "medium",
            "reasoning": "Standard taxi clearance from Terminal B to runway 22R",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "standard_taxi_clearance",
                "severity": "low",
                "route": "B-A-A7",
                "estimated_taxi_time": "8 minutes",
                "runway": "22R",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Standard taxi clearances should use most efficient available route",
        }

    async def _create_hold_for_crossing_traffic(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a hold for crossing traffic scenario"""

        aircraft_states = {"all_aircraft": []}

        # Aircraft needing to cross active taxiway
        crossing_callsign = f"{random.choice(['AAL', 'DAL'])}{random.randint(100, 999)}"
        crossing_aircraft = {
            "callsign": crossing_callsign,
            "aircraft_type": "A319",
            "lat": 40.6418,
            "lon": -73.7788,
            "altitude": "ground",
            "speed": 0.0,
            "heading": 180,
            "flight_phase": "holding_for_crossing",
            "runway_proximity": "taxiway_intersection",
            "airport_area": "central_field",
            "destination_gate": "C5",
            "crossing_point": "Taxiway A",
        }

        # Active traffic on the taxiway
        active_callsign = f"{random.choice(['JBU', 'UAL'])}{random.randint(100, 999)}"
        active_aircraft = {
            "callsign": active_callsign,
            "aircraft_type": "B752",
            "lat": 40.6415,
            "lon": -73.7792,
            "altitude": "ground",
            "speed": 18.0,
            "heading": 270,
            "flight_phase": "taxi_to_runway",
            "runway_proximity": "taxiway_A",
            "airport_area": "central_field",
            "destination_runway": "04L",
        }

        aircraft_states["all_aircraft"] = [crossing_aircraft, active_aircraft]

        expected_action = {
            "command_type": "crossing_control",
            "details": f"{crossing_callsign} hold short of Taxiway A for crossing traffic",
            "affected_aircraft": [crossing_callsign],
            "priority": "high",
            "reasoning": "Must wait for active taxiway traffic to clear before crossing",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "hold_for_crossing",
                "severity": "medium",
                "crossing_point": "Taxiway A",
                "active_traffic": active_callsign,
                "estimated_wait": "1 minute",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Active taxiway traffic has right of way over crossing traffic",
        }

    async def _create_taxi_priority_emergency(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a taxi priority for emergency scenario"""

        aircraft_states = {"all_aircraft": []}

        # Emergency aircraft needing priority taxi
        emergency_callsign = f"UAL{random.randint(100, 999)}"
        emergency_aircraft = {
            "callsign": emergency_callsign,
            "aircraft_type": "B777",
            "lat": 40.6410,
            "lon": -73.7795,
            "altitude": "ground",
            "speed": 5.0,
            "heading": 180,
            "flight_phase": "emergency_taxi",
            "runway_proximity": "taxiway_C",
            "airport_area": "north_field",
            "emergency_type": "medical",
            "priority_level": "urgent",
            "destination_gate": "Medical Bay",
        }

        # Normal traffic that needs to give way
        normal_callsign = f"AAL{random.randint(100, 999)}"
        normal_aircraft = {
            "callsign": normal_callsign,
            "aircraft_type": "A321",
            "lat": 40.6412,
            "lon": -73.7790,
            "altitude": "ground",
            "speed": 10.0,
            "heading": 270,
            "flight_phase": "taxi_to_runway",
            "runway_proximity": "taxiway_A",
            "airport_area": "central_field",
            "destination_runway": "22L",
        }

        aircraft_states["all_aircraft"] = [emergency_aircraft, normal_aircraft]

        expected_action = {
            "command_type": "emergency_priority",
            "details": f"{normal_callsign} hold position, emergency aircraft {emergency_callsign} has priority",
            "affected_aircraft": [normal_callsign, emergency_callsign],
            "priority": "urgent",
            "reasoning": "Medical emergency requires immediate priority routing to medical bay",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "emergency_priority_taxi",
                "severity": "urgent",
                "emergency_type": "medical",
                "affected_normal_traffic": 1,
            },
            "expected_atc_action": expected_action,
            "training_notes": "Emergency aircraft always have absolute priority over normal traffic",
        }

    async def _create_hold_weather_delay(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Create a hold for weather delay scenario"""

        aircraft_states = {"all_aircraft": []}

        # Multiple aircraft holding due to weather
        for i in range(3):
            callsign = f"{random.choice(['AAL', 'DAL', 'JBU'])}{random.randint(100, 999)}"
            aircraft = {
                "callsign": callsign,
                "aircraft_type": random.choice(["B738", "A320"]),
                "lat": 40.6420 + (i * 0.002),
                "lon": -73.7785 + (i * 0.002),
                "altitude": "ground",
                "speed": 0.0,
                "heading": 220,
                "flight_phase": "holding_weather",
                "runway_proximity": f"holding_area_{i+1}",
                "airport_area": "departure_area",
                "destination": random.choice(["KORD", "KBOS", "KMIA"]),
                "hold_reason": "thunderstorms",
                "fuel_remaining_minutes": random.randint(90, 180),
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "weather_hold",
            "details": "All aircraft continue holding, runway operations suspended due to thunderstorms",
            "affected_aircraft": [a["callsign"] for a in aircraft_states["all_aircraft"]],
            "priority": "high",
            "reasoning": "Safety requires suspension of operations during active thunderstorms",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "weather_hold",
                "severity": "high",
                "weather_condition": "thunderstorms",
                "estimated_delay": "20 minutes",
                "affected_aircraft": 3,
            },
            "expected_atc_action": expected_action,
            "training_notes": "Weather safety takes absolute priority over operational efficiency",
        }

    async def _create_taxi_congestion_alley(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Create a taxi congestion in alley scenario"""

        aircraft_states = {"all_aircraft": []}

        # Multiple aircraft in narrow taxiway creating congestion
        for i in range(4):
            callsign = f"{random.choice(['AAL', 'DAL', 'JBU', 'UAL'])}{random.randint(100, 999)}"
            aircraft = {
                "callsign": callsign,
                "aircraft_type": "A320",
                "lat": 40.6425 - (i * 0.001),
                "lon": -73.7795,
                "altitude": "ground",
                "speed": 2.0 if i == 0 else 0.0,  # Only lead aircraft moving
                "heading": 180,
                "flight_phase": "taxi_congestion",
                "runway_proximity": "taxiway_alley",
                "airport_area": "terminal_connector",
                "destination": "various_gates" if i < 2 else "runway_22L",
                "congestion_position": i + 1,
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "congestion_management",
            "details": f"{aircraft_states['all_aircraft'][0]['callsign']} expedite taxi, following aircraft maintain spacing",
            "affected_aircraft": [aircraft_states["all_aircraft"][0]["callsign"]],
            "priority": "medium",
            "reasoning": "Clear congestion by expediting lead aircraft movement",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "taxi_alley_congestion",
                "severity": "medium",
                "congested_taxiway": "Terminal Connector",
                "aircraft_count": 4,
                "estimated_clear_time": "5 minutes",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Congestion requires expediting lead traffic to clear bottlenecks",
        }

    async def _create_hold_maintenance_blocking(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a hold for maintenance blocking scenario"""

        aircraft_states = {"all_aircraft": []}

        # Aircraft needing to hold due to maintenance vehicle
        callsign = f"{random.choice(['JBU', 'SWA'])}{random.randint(100, 999)}"
        aircraft = {
            "callsign": callsign,
            "aircraft_type": "B737",
            "lat": 40.6415,
            "lon": -73.7790,
            "altitude": "ground",
            "speed": 0.0,
            "heading": 90,
            "flight_phase": "holding_for_maintenance",
            "runway_proximity": "taxiway_B",
            "airport_area": "maintenance_area",
            "destination_runway": "04R",
            "hold_reason": "maintenance_vehicle_blocking",
            "maintenance_activity": "runway_inspection",
        }

        aircraft_states["all_aircraft"] = [aircraft]

        expected_action = {
            "command_type": "maintenance_hold",
            "details": f"{callsign} hold position, maintenance vehicle active on Taxiway B",
            "affected_aircraft": [callsign],
            "priority": "medium",
            "reasoning": "Maintenance operations require aircraft to hold until area is clear",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "maintenance_blocking",
                "severity": "medium",
                "blocked_area": "Taxiway B",
                "maintenance_type": "runway_inspection",
                "estimated_clear_time": "10 minutes",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Maintenance operations have priority, aircraft must hold until clear",
        }

    async def _create_generic_taxi_hold_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a generic taxi/hold scenario"""
        return {
            **base,
            "aircraft_states": {"all_aircraft": []},
            "bottleneck_info": {"type": "generic", "severity": "low"},
            "expected_atc_action": {
                "command_type": "monitor",
                "details": "Monitor taxi operations",
            },
            "training_notes": "Generic scenario for baseline comparison",
        }

    async def save_scenarios(
        self, scenarios: List[Dict[str, Any]], filename: str = None
    ):
        """Save scenarios to JSON file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"taxi_hold_scenarios_{timestamp}.json"

        filepath = os.path.join(self.scenarios_dir, filename)

        with open(filepath, "w") as f:
            json.dump(scenarios, f, indent=2)

        print(f"💾 Saved {len(scenarios)} taxi & hold scenarios to: {filename}")
        return filepath

    async def validate_scenarios(
        self, scenarios: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate scenario quality using CleaningAgent"""

        print("🔍 Validating scenario quality...")

        validation_results = {
            "total_scenarios": len(scenarios),
            "high_quality": 0,
            "medium_quality": 0,
            "low_quality": 0,
            "detailed_results": [],
        }

        for i, scenario in enumerate(scenarios):
            if i % 10 == 0:
                print(f"   Validated {i}/{len(scenarios)} scenarios...")

            # Use CleaningAgent to assess scenario quality
            result = await self.cleaning_agent.process({"validation_record": scenario})
            quality_score = result.get("quality_score", 0.5)

            if quality_score >= 0.8:
                validation_results["high_quality"] += 1
            elif quality_score >= 0.6:
                validation_results["medium_quality"] += 1
            else:
                validation_results["low_quality"] += 1

            validation_results["detailed_results"].append(
                {
                    "scenario_id": scenario.get("record_id"),
                    "scenario_type": scenario.get("scenario_type"),
                    "quality_score": quality_score,
                }
            )

        return validation_results


async def main():
    """Main entry point for taxi & hold data curator"""

    parser = argparse.ArgumentParser(
        description="Taxi & Hold Data Curator - Create specialized taxi and hold scenarios",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
    python3.9 taxi_hold_curator.py                    # Generate 50 scenarios
    python3.9 taxi_hold_curator.py --scenarios 100   # Generate 100 scenarios
    python3.9 taxi_hold_curator.py --validate-only   # Only validate existing scenarios
        """,
    )

    parser.add_argument(
        "--scenarios",
        type=int,
        default=50,
        help="Number of scenarios to generate (default: 50)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output filename (default: auto-generated with timestamp)",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing scenarios, don't generate new ones",
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = get_config()

        # Initialize curator
        curator = TaxiHoldScenarioCurator(
            cerebras_api_key=config["cerebras_api_key"],
            base_path=config["paths"]["adas"],
        )

        if not args.validate_only:
            # Generate scenarios
            scenarios = await curator.generate_taxi_hold_scenarios(args.scenarios)

            # Save scenarios
            filepath = await curator.save_scenarios(scenarios, args.output_file)

            # Validate scenarios
            validation_results = await curator.validate_scenarios(scenarios)

            # Print results
            print("\n" + "=" * 60)
            print("🏆 TAXI & HOLD SCENARIO GENERATION COMPLETE")
            print("=" * 60)
            print(f"📊 Generated: {validation_results['total_scenarios']} scenarios")
            print(f"🟢 High Quality: {validation_results['high_quality']}")
            print(f"🟡 Medium Quality: {validation_results['medium_quality']}")
            print(f"🔴 Low Quality: {validation_results['low_quality']}")
            print(f"💾 Saved to: {os.path.basename(filepath)}")

        else:
            print("🔍 Validation-only mode not implemented yet")

    except KeyboardInterrupt:
        print("\n⏹️  Generation interrupted by user")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
