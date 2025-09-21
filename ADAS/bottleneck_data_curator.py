#!/usr/bin/env python3
"""
Bottleneck Data Curator - Specialized Tool for Creating Bottleneck Resolution Datasets

This script creates synthetic and curated bottleneck scenarios specifically for training
the ADAS TaskAgent to handle runway congestion, approach delays, ground traffic conflicts,
and other ATC bottleneck situations.

Usage:
    python3.9 bottleneck_data_curator.py
    python3.9 bottleneck_data_curator.py --scenarios 50 --output-file bottleneck_scenarios.json
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, List
import random
import uuid

from cleaning_agent import CleaningAgent
from config import get_config


class BottleneckScenarioCurator:
    """Creates curated bottleneck resolution scenarios for ADAS training"""

    def __init__(self, cerebras_api_key: str, base_path: str):
        self.cleaning_agent = CleaningAgent(cerebras_api_key)
        self.base_path = base_path
        self.scenarios_dir = os.path.join(base_path, "bottleneck-scenarios")
        os.makedirs(self.scenarios_dir, exist_ok=True)

    async def generate_bottleneck_scenarios(
        self, num_scenarios: int = 50
    ) -> List[Dict[str, Any]]:
        """Generate synthetic bottleneck scenarios for training"""

        scenarios = []
        scenario_types = [
            "runway_congestion",
            "approach_delays",
            "ground_traffic_conflict",
            "gate_area_bottleneck",
            "weather_impact",
            "equipment_failure",
            "multiple_departures",
            "arrival_rush",
        ]

        print(f"🏭 Generating {num_scenarios} bottleneck scenarios...")

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
        """Create a specific type of bottleneck scenario"""

        base_scenario = {
            "record_id": str(uuid.uuid4()),
            "scenario_type": scenario_type,
            "scenario_number": scenario_num,
            "timestamp": datetime.now().isoformat(),
            "airport": "KJFK",
        }

        if scenario_type == "runway_congestion":
            return await self._create_runway_congestion_scenario(base_scenario)
        elif scenario_type == "approach_delays":
            return await self._create_approach_delay_scenario(base_scenario)
        elif scenario_type == "ground_traffic_conflict":
            return await self._create_ground_conflict_scenario(base_scenario)
        elif scenario_type == "gate_area_bottleneck":
            return await self._create_gate_bottleneck_scenario(base_scenario)
        elif scenario_type == "weather_impact":
            return await self._create_weather_scenario(base_scenario)
        elif scenario_type == "equipment_failure":
            return await self._create_equipment_failure_scenario(base_scenario)
        elif scenario_type == "multiple_departures":
            return await self._create_multiple_departures_scenario(base_scenario)
        elif scenario_type == "arrival_rush":
            return await self._create_arrival_rush_scenario(base_scenario)
        else:
            return await self._create_generic_bottleneck_scenario(base_scenario)

    async def _create_runway_congestion_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a runway congestion bottleneck scenario"""

        # Generate 4-6 aircraft queued for same runway
        aircraft_count = random.randint(4, 6)
        runway = random.choice(["22L", "22R", "04L", "04R"])

        aircraft_states = {"all_aircraft": []}

        for i in range(aircraft_count):
            callsign = f"{random.choice(['AAL', 'DAL', 'JBU', 'UAL'])}{random.randint(100, 999)}"

            # Position aircraft in runway queue
            base_lat = 40.6413 + (i * 0.002)  # Spaced along taxiway
            base_lon = -73.7781 + (i * 0.001)

            aircraft = {
                "callsign": callsign,
                "aircraft_type": random.choice(["B738", "A320", "B752", "A321"]),
                "lat": base_lat,
                "lon": base_lon,
                "altitude": "ground",
                "speed": 0.0 if i > 0 else 5.0,  # Lead aircraft moving slowly
                "heading": 220,
                "flight_phase": "taxi_to_runway" if i == 0 else "holding",
                "runway_proximity": f"queued_for_{runway}",
                "airport_area": "runway_approach",
                "fuel_remaining_minutes": random.randint(45, 120),
                "passengers": random.randint(120, 180),
                "destination": random.choice(["KLAX", "KORD", "KBOS", "KMIA"]),
            }
            aircraft_states["all_aircraft"].append(aircraft)

        # Expected ATC resolution
        expected_action = {
            "command_type": "runway_management",
            "details": f"Expedite {aircraft_states['all_aircraft'][0]['callsign']} departure, hold remaining aircraft",
            "affected_aircraft": [aircraft_states["all_aircraft"][0]["callsign"]],
            "runway": runway,
            "priority": "high",
            "reasoning": "Runway congestion requires immediate departure clearance to prevent extended delays",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "runway_congestion",
                "severity": "high",
                "affected_runway": runway,
                "queue_length": aircraft_count,
                "estimated_delay": aircraft_count * 2,  # 2 minutes per aircraft
            },
            "expected_atc_action": expected_action,
            "training_notes": "TaskAgent should prioritize runway throughput while maintaining safety separation",
        }

    async def _create_approach_delay_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an approach delay bottleneck scenario"""

        aircraft_count = random.randint(3, 5)
        runway = random.choice(["22L", "22R"])

        aircraft_states = {"all_aircraft": []}

        for i in range(aircraft_count):
            callsign = f"{random.choice(['AAL', 'DAL', 'JBU', 'UAL'])}{random.randint(100, 999)}"

            # Position aircraft on approach
            distance_out = 15 - (i * 3)  # 15, 12, 9, 6, 3 miles out
            base_lat = 40.6413 - (distance_out * 0.01)
            base_lon = -73.7781 - (distance_out * 0.005)

            aircraft = {
                "callsign": callsign,
                "aircraft_type": random.choice(["B738", "A320", "B752"]),
                "lat": base_lat,
                "lon": base_lon,
                "altitude": 3000 - (i * 500),
                "speed": 180 - (i * 10),  # Decreasing speed in sequence
                "heading": 220,
                "flight_phase": "approach",
                "runway_proximity": f"approach_{runway}",
                "airport_area": "approach_corridor",
                "fuel_remaining_minutes": random.randint(25, 60),
                "distance_to_runway": distance_out,
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "approach_spacing",
            "details": f"Vector {aircraft_states['all_aircraft'][0]['callsign']} for spacing, reduce speed to final approach",
            "affected_aircraft": [aircraft_states["all_aircraft"][0]["callsign"]],
            "runway": runway,
            "priority": "medium",
            "reasoning": "Approach spacing needed to maintain safe separation and prevent go-around",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "approach_congestion",
                "severity": "medium",
                "affected_runway": runway,
                "aircraft_in_sequence": aircraft_count,
                "minimum_separation": "3 miles",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Focus on maintaining approach separation while maximizing runway utilization",
        }

    async def _create_ground_conflict_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a ground traffic conflict scenario"""

        # Two aircraft on intersecting taxiways
        aircraft_states = {"all_aircraft": []}

        # Aircraft 1: On Taxiway A
        callsign1 = f"{random.choice(['AAL', 'JBU'])}{random.randint(100, 999)}"
        aircraft1 = {
            "callsign": callsign1,
            "aircraft_type": "B738",
            "lat": 40.6420,
            "lon": -73.7790,
            "altitude": "ground",
            "speed": 15.0,
            "heading": 90,
            "flight_phase": "taxi_to_gate",
            "runway_proximity": "taxiway_A",
            "airport_area": "north_field",
        }

        # Aircraft 2: On intersecting Taxiway B
        callsign2 = f"{random.choice(['DAL', 'UAL'])}{random.randint(100, 999)}"
        aircraft2 = {
            "callsign": callsign2,
            "aircraft_type": "A320",
            "lat": 40.6415,
            "lon": -73.7785,
            "altitude": "ground",
            "speed": 12.0,
            "heading": 180,
            "flight_phase": "taxi_to_runway",
            "runway_proximity": "taxiway_B",
            "airport_area": "central_field",
        }

        aircraft_states["all_aircraft"] = [aircraft1, aircraft2]

        expected_action = {
            "command_type": "ground_control",
            "details": f"{callsign1} hold short of Taxiway B, give way to {callsign2}",
            "affected_aircraft": [callsign1, callsign2],
            "priority": "high",
            "reasoning": "Prevent ground collision at taxiway intersection",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "ground_conflict",
                "severity": "high",
                "conflict_point": "Taxiway A/B intersection",
                "risk_level": "collision_risk",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Priority should be given based on aircraft destination and traffic flow efficiency",
        }

    async def _create_gate_bottleneck_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a gate area bottleneck scenario"""

        aircraft_states = {"all_aircraft": []}

        # Multiple aircraft competing for limited gates
        for i in range(3):
            callsign = (
                f"{random.choice(['AAL', 'DAL', 'JBU'])}{random.randint(100, 999)}"
            )
            aircraft = {
                "callsign": callsign,
                "aircraft_type": random.choice(["B738", "A320"]),
                "lat": 40.6400 + (i * 0.001),
                "lon": -73.7800 + (i * 0.001),
                "altitude": "ground",
                "speed": 5.0,
                "heading": 45,
                "flight_phase": "taxi_to_gate",
                "runway_proximity": "gate_area",
                "airport_area": "terminal_area",
                "gate_assignment": f"A{10 + i}" if i < 2 else "unassigned",
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "gate_assignment",
            "details": f"Assign {aircraft_states['all_aircraft'][2]['callsign']} to gate A13",
            "affected_aircraft": [aircraft_states["all_aircraft"][2]["callsign"]],
            "priority": "medium",
            "reasoning": "Resolve gate conflict by reassigning aircraft to available gate",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "gate_bottleneck",
                "severity": "medium",
                "available_gates": 1,
                "waiting_aircraft": 1,
            },
            "expected_atc_action": expected_action,
            "training_notes": "Gate assignments should optimize passenger flow and aircraft turnaround time",
        }

    async def _create_weather_scenario(self, base: Dict[str, Any]) -> Dict[str, Any]:
        """Create a weather impact bottleneck scenario"""

        aircraft_states = {"all_aircraft": []}

        # Aircraft holding due to weather
        for i in range(4):
            callsign = f"{random.choice(['AAL', 'DAL', 'JBU', 'UAL'])}{random.randint(100, 999)}"
            aircraft = {
                "callsign": callsign,
                "aircraft_type": random.choice(["B738", "A320", "B752"]),
                "lat": 40.6413 + (i * 0.01),
                "lon": -73.7781 + (i * 0.01),
                "altitude": 8000 + (i * 1000),
                "speed": 200,
                "heading": random.randint(0, 360),
                "flight_phase": "holding",
                "runway_proximity": "holding_pattern",
                "airport_area": "approach_airspace",
                "fuel_remaining_minutes": 45 - (i * 5),  # Decreasing fuel
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "priority_landing",
            "details": f"Clear {aircraft_states['all_aircraft'][3]['callsign']} for immediate approach due to low fuel",
            "affected_aircraft": [aircraft_states["all_aircraft"][3]["callsign"]],
            "priority": "urgent",
            "reasoning": "Aircraft has minimum fuel reserves, requires priority handling",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "weather_delay",
                "severity": "high",
                "weather_condition": "thunderstorms",
                "runway_closure_time": "15 minutes",
                "aircraft_in_holding": 4,
            },
            "expected_atc_action": expected_action,
            "training_notes": "Weather delays require fuel-based prioritization and efficient sequencing",
        }

    async def _create_equipment_failure_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an equipment failure bottleneck scenario"""

        aircraft_states = {"all_aircraft": []}

        # Aircraft with emergency blocking runway
        emergency_callsign = f"UAL{random.randint(100, 999)}"
        emergency_aircraft = {
            "callsign": emergency_callsign,
            "aircraft_type": "B757",
            "lat": 40.6413,
            "lon": -73.7781,
            "altitude": "ground",
            "speed": 0.0,
            "heading": 220,
            "flight_phase": "emergency_stopped",
            "runway_proximity": "on_runway_22L",
            "airport_area": "runway",
            "emergency_type": "hydraulic_failure",
        }
        aircraft_states["all_aircraft"].append(emergency_aircraft)

        # Other aircraft affected by runway closure
        for i in range(3):
            callsign = (
                f"{random.choice(['AAL', 'DAL', 'JBU'])}{random.randint(100, 999)}"
            )
            aircraft = {
                "callsign": callsign,
                "aircraft_type": "A320",
                "lat": 40.6413 + (i * 0.002),
                "lon": -73.7781 + (i * 0.002),
                "altitude": "ground",
                "speed": 0.0,
                "heading": 220,
                "flight_phase": "holding",
                "runway_proximity": "queued_for_22L",
                "airport_area": "taxiway",
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "runway_diversion",
            "details": "Divert all departures to runway 22R, emergency services to runway 22L",
            "affected_aircraft": [
                a["callsign"] for a in aircraft_states["all_aircraft"][1:]
            ],
            "priority": "urgent",
            "reasoning": "Runway 22L blocked by emergency, divert traffic to alternate runway",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "equipment_failure",
                "severity": "urgent",
                "blocked_runway": "22L",
                "alternate_runway": "22R",
                "emergency_services": "responding",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Emergency situations require immediate runway diversions and clear communication",
        }

    async def _create_multiple_departures_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a multiple departures bottleneck scenario"""

        aircraft_states = {"all_aircraft": []}

        # Multiple aircraft ready for departure at same time
        for i in range(5):
            callsign = f"{random.choice(['AAL', 'DAL', 'JBU', 'UAL', 'SWA'])}{random.randint(100, 999)}"
            aircraft = {
                "callsign": callsign,
                "aircraft_type": random.choice(["B738", "A320", "B752"]),
                "lat": 40.6413 + (i * 0.001),
                "lon": -73.7781 + (i * 0.001),
                "altitude": "ground",
                "speed": 0.0,
                "heading": 220,
                "flight_phase": "ready_for_departure",
                "runway_proximity": f"holding_point_{i+1}",
                "airport_area": "departure_queue",
                "departure_time": (datetime.now() + timedelta(minutes=i)).isoformat(),
                "destination": random.choice(["KLAX", "KORD", "KBOS", "KMIA", "KDEN"]),
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "departure_sequencing",
            "details": f"Clear {aircraft_states['all_aircraft'][0]['callsign']} for immediate departure, sequence others by destination",
            "affected_aircraft": [
                a["callsign"] for a in aircraft_states["all_aircraft"]
            ],
            "priority": "high",
            "reasoning": "Optimize departure sequence based on routing and minimize delays",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "departure_rush",
                "severity": "high",
                "aircraft_ready": 5,
                "runway_capacity": "2 per minute",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Departure sequencing should consider routing efficiency and slot times",
        }

    async def _create_arrival_rush_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create an arrival rush bottleneck scenario"""

        aircraft_states = {"all_aircraft": []}

        # Multiple aircraft arriving simultaneously
        for i in range(6):
            callsign = f"{random.choice(['AAL', 'DAL', 'JBU', 'UAL'])}{random.randint(100, 999)}"
            distance = 25 - (i * 3)  # 25, 22, 19, 16, 13, 10 miles out

            aircraft = {
                "callsign": callsign,
                "aircraft_type": random.choice(["B738", "A320", "B752"]),
                "lat": 40.6413 - (distance * 0.008),
                "lon": -73.7781 - (distance * 0.004),
                "altitude": 4000 - (i * 200),
                "speed": 220 - (i * 5),
                "heading": 220,
                "flight_phase": "approach",
                "runway_proximity": "approach_22L",
                "airport_area": "approach_corridor",
                "distance_to_runway": distance,
                "fuel_remaining_minutes": random.randint(30, 90),
            }
            aircraft_states["all_aircraft"].append(aircraft)

        expected_action = {
            "command_type": "approach_management",
            "details": f"Vector {aircraft_states['all_aircraft'][2]['callsign']} to extend downwind, maintain spacing",
            "affected_aircraft": [aircraft_states["all_aircraft"][2]["callsign"]],
            "priority": "high",
            "reasoning": "Extend approach to maintain safe separation in arrival rush",
        }

        return {
            **base,
            "aircraft_states": aircraft_states,
            "bottleneck_info": {
                "type": "arrival_rush",
                "severity": "high",
                "aircraft_in_sequence": 6,
                "runway_rate": "1 per 2 minutes",
                "separation_required": "3 miles",
            },
            "expected_atc_action": expected_action,
            "training_notes": "Arrival rushes require careful spacing and may need vectoring or speed control",
        }

    async def _create_generic_bottleneck_scenario(
        self, base: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a generic bottleneck scenario"""
        return {
            **base,
            "aircraft_states": {"all_aircraft": []},
            "bottleneck_info": {"type": "generic", "severity": "low"},
            "expected_atc_action": {
                "command_type": "no_action",
                "details": "Monitor situation",
            },
            "training_notes": "Generic scenario for baseline comparison",
        }

    async def save_scenarios(
        self, scenarios: List[Dict[str, Any]], filename: str = None
    ):
        """Save scenarios to JSON file"""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bottleneck_scenarios_{timestamp}.json"

        filepath = os.path.join(self.scenarios_dir, filename)

        with open(filepath, "w") as f:
            json.dump(scenarios, f, indent=2)

        print(f"💾 Saved {len(scenarios)} bottleneck scenarios to: {filename}")
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
    """Main entry point for bottleneck data curator"""

    parser = argparse.ArgumentParser(
        description="Bottleneck Data Curator - Create specialized ATC bottleneck scenarios",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""Examples:
    python3.9 bottleneck_data_curator.py                    # Generate 50 scenarios
    python3.9 bottleneck_data_curator.py --scenarios 100   # Generate 100 scenarios
    python3.9 bottleneck_data_curator.py --validate-only   # Only validate existing scenarios
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
        curator = BottleneckScenarioCurator(
            cerebras_api_key=config["cerebras_api_key"],
            base_path=config["paths"]["adas"],
        )

        if not args.validate_only:
            # Generate scenarios
            scenarios = await curator.generate_bottleneck_scenarios(args.scenarios)

            # Save scenarios
            filepath = await curator.save_scenarios(scenarios, args.output_file)

            # Validate scenarios
            validation_results = await curator.validate_scenarios(scenarios)

            # Print results
            print("\n" + "=" * 60)
            print("🏆 BOTTLENECK SCENARIO GENERATION COMPLETE")
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
