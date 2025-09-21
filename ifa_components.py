"""
Core components for Informed Fast ATC Transcriber (IFA)
"""

import re
import time
import json
import os
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict

try:
    from client import PlaneMonitor
except ImportError:
    print("Warning: client.py not available")


@dataclass
class ValidationRecord:
    """Structure for validation dataset records"""

    record_id: str
    timestamp_speech_start: str
    timestamp_processing_complete: str
    processing_lag_seconds: float
    aircraft_states: Dict[str, Any]
    atc_command: Dict[str, Any]
    correlation_metadata: Dict[str, Any]


class ATCCommandParser:
    """Parse and categorize ATC commands with JFK-specific knowledge"""

    # Command type patterns
    COMMAND_TYPES = {
        "taxi": ["taxi", "proceed via", "continue taxi", "taxi via"],
        "takeoff": ["cleared for takeoff", "line up and wait", "cleared takeoff"],
        "landing": ["cleared to land", "cleared land", "go around", "missed approach"],
        "frequency_change": ["contact", "monitor", "switch to", "over to"],
        "hold": ["hold short", "hold position", "standby", "hold"],
        "runway_crossing": ["cross runway", "cross", "expedite crossing"],
        "altitude": ["climb", "descend", "maintain altitude", "flight level"],
        "speed": ["reduce speed", "increase speed", "slow to", "maintain speed"],
        "acknowledgment": ["roger", "wilco", "affirmative", "copy", "understood"],
    }

    # JFK-specific runway patterns
    JFK_RUNWAYS = ["04L", "04R", "22L", "22R", "13L", "13R", "31L", "31R"]

    # JFK-specific taxiway patterns
    JFK_TAXIWAYS = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "J",
        "K",
        "L",
        "M",
        "N",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "AA",
        "BB",
        "CC",
        "DD",
        "EE",
        "FF",
        "GG",
        "HH",
        "JJ",
        "KK",
    ]

    def __init__(self):
        self.callsign_pattern = re.compile(
            r"\b[A-Z]{2,3}\s*\d{1,4}[A-Z]?\b|\b[A-Z]\d{2,4}\b"
        )
        self.frequency_pattern = re.compile(r"\b1\d{2}\.\d{1,2}\b")
        self.runway_pattern = re.compile(
            r"\b(?:runway\s*)?(" + "|".join(self.JFK_RUNWAYS) + r")\b", re.IGNORECASE
        )
        self.taxiway_pattern = re.compile(
            r"\b(?:taxiway\s*)?(" + "|".join(self.JFK_TAXIWAYS) + r")\b", re.IGNORECASE
        )

    def parse_command(self, transcription: str, explanation: str) -> Dict[str, Any]:
        """Parse ATC command and extract structured information"""

        # Determine command type
        command_type = self._categorize_command(transcription.lower())

        # Extract elements
        callsigns = self._extract_callsigns(transcription)
        frequencies = self._extract_frequencies(transcription)
        runways = self._extract_runways(transcription)
        taxiways = self._extract_taxiways(transcription)

        # Assess confidence based on transcription clarity
        confidence = self._assess_confidence(transcription, explanation)

        return {
            "raw_transcription": transcription,
            "processed_explanation": explanation,
            "command_type": command_type,
            "confidence_score": confidence,
            "affected_aircraft": callsigns,
            "extracted_elements": {
                "callsigns": callsigns,
                "frequencies": frequencies,
                "runways": runways,
                "taxiways": taxiways,
            },
        }

    def _categorize_command(self, text: str) -> str:
        """Categorize command type based on keywords"""
        for cmd_type, keywords in self.COMMAND_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return cmd_type
        return "unknown"

    def _extract_callsigns(self, text: str) -> List[str]:
        """Extract aircraft callsigns from transcription"""
        matches = self.callsign_pattern.findall(text.upper())
        return [match.replace(" ", "") for match in matches]

    def _extract_frequencies(self, text: str) -> List[str]:
        """Extract radio frequencies from transcription"""
        return self.frequency_pattern.findall(text)

    def _extract_runways(self, text: str) -> List[str]:
        """Extract JFK runway identifiers"""
        matches = self.runway_pattern.findall(text)
        return [match.upper() for match in matches]

    def _extract_taxiways(self, text: str) -> List[str]:
        """Extract JFK taxiway identifiers"""
        matches = self.taxiway_pattern.findall(text)
        return [match.upper() for match in matches]

    def _assess_confidence(self, transcription: str, explanation: str) -> float:
        """Assess transcription confidence based on clarity indicators"""
        confidence = 1.0

        # Reduce confidence for unclear indicators
        if any(
            indicator in explanation.lower()
            for indicator in [
                "unclear",
                "misheard",
                "static",
                "overlapping",
                "uncertain",
            ]
        ):
            confidence -= 0.3

        # Reduce confidence for very short transcriptions
        if len(transcription.split()) < 3:
            confidence -= 0.2

        # Reduce confidence for missing callsigns in commands that should have them
        if (
            not self._extract_callsigns(transcription)
            and "contact" not in transcription.lower()
        ):
            confidence -= 0.2

        return max(0.1, confidence)


class AircraftStateManager:
    """Manage aircraft state collection synchronized with transcription events"""

    def __init__(self):
        try:
            self.plane_monitor = PlaneMonitor()
        except:
            print("Warning: PlaneMonitor not available")
            self.plane_monitor = None
        self.current_state = None
        self.last_update = None

    async def get_current_aircraft_state(self) -> Dict[str, Any]:
        """Fetch current aircraft state from JFK area"""
        if not self.plane_monitor:
            return self._get_empty_state()

        try:
            # Get fresh aircraft data
            aircraft_data = await self.plane_monitor.fetch_planes()
            self.current_state = aircraft_data
            self.last_update = time.time()

            # Focus on ground aircraft at JFK (expandable)
            ground_aircraft = aircraft_data.get("current_planes", {}).get("ground", {})
            air_aircraft = aircraft_data.get("current_planes", {}).get("air", {})

            # TODO: Expand to include approach/departure aircraft
            # Filters could include:
            # - Aircraft below 3000ft altitude
            # - Aircraft within 10nm of JFK
            # - Aircraft on specific approach/departure routes

            # Calculate runway occupancy (basic implementation)
            runway_occupancy = self._calculate_runway_occupancy(ground_aircraft)

            # Add flight numbers as callsign field and enhance with contextual data
            ground_aircraft_with_callsigns = []
            for flight_number, aircraft_data_item in ground_aircraft.items():
                aircraft_with_callsign = aircraft_data_item.copy()
                aircraft_with_callsign["callsign"] = flight_number
                # Add contextual enhancements
                aircraft_with_callsign = self._enhance_aircraft_context(
                    aircraft_with_callsign
                )
                ground_aircraft_with_callsigns.append(aircraft_with_callsign)

            air_aircraft_with_callsigns = []
            for flight_number, aircraft_data_item in air_aircraft.items():
                aircraft_with_callsign = aircraft_data_item.copy()
                aircraft_with_callsign["callsign"] = flight_number
                # Add contextual enhancements
                aircraft_with_callsign = self._enhance_aircraft_context(
                    aircraft_with_callsign
                )
                air_aircraft_with_callsigns.append(aircraft_with_callsign)

            return {
                "all_aircraft": ground_aircraft_with_callsigns
                + air_aircraft_with_callsigns,
                "jfk_ground_aircraft": ground_aircraft_with_callsigns,
                "jfk_air_aircraft": air_aircraft_with_callsigns,  # Currently within 5nm
                "runway_occupancy": runway_occupancy,
                "total_aircraft_count": aircraft_data.get("total_aircraft", 0),
                "timestamp": aircraft_data.get("timestamp", datetime.now().isoformat()),
            }

        except Exception as e:
            print(f"❌ Error fetching aircraft state: {e}")
            return self._get_empty_state(error=str(e))

    def _enhance_aircraft_context(self, aircraft: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance aircraft data with contextual information for transcription"""
        enhanced = aircraft.copy()

        # Aircraft type context
        aircraft_type = aircraft.get("aircraft_type", "Unknown")
        enhanced["aircraft_type_description"] = self._get_aircraft_type_description(
            aircraft_type
        )

        # Flight phase detection
        altitude = aircraft.get("altitude", 0)
        speed = aircraft.get("speed", 0)

        try:
            altitude_num = (
                float(altitude) if altitude != "N/A" and altitude is not None else 0
            )
            speed_num = float(speed) if speed != "N/A" and speed is not None else 0
        except (ValueError, TypeError):
            altitude_num = 0
            speed_num = 0

        enhanced["flight_phase"] = self._detect_flight_phase(altitude_num, speed_num)

        # Position context relative to JFK
        lat = aircraft.get("lat")
        lon = aircraft.get("lon")
        if lat is not None and lon is not None and lat != "N/A" and lon != "N/A":
            # The helper methods now handle None values internally
            enhanced["runway_proximity"] = self._calculate_runway_proximity(lat, lon)
            enhanced["airport_area"] = self._determine_airport_area(lat, lon)
        else:
            enhanced["runway_proximity"] = "no_position_data"
            enhanced["airport_area"] = "no_position_data"

        return enhanced

    def _get_aircraft_type_description(self, aircraft_type: str) -> str:
        """Get human-readable aircraft type description"""
        # Common aircraft types at JFK
        aircraft_types = {
            "A359": "Airbus A350-900 (wide-body, long-haul)",
            "A21N": "Airbus A321neo (narrow-body)",
            "B738": "Boeing 737-800 (narrow-body)",
            "B77W": "Boeing 777-300ER (wide-body, long-haul)",
            "B789": "Boeing 787-9 (wide-body, long-haul)",
            "A388": "Airbus A380 (double-deck, very heavy)",
            "B748": "Boeing 747-8 (wide-body, very heavy)",
            "A333": "Airbus A330-300 (wide-body)",
            "B763": "Boeing 767-300 (wide-body)",
            "A320": "Airbus A320 (narrow-body)",
            "B737": "Boeing 737 (narrow-body)",
            "E190": "Embraer E190 (regional jet)",
            "CRJ9": "Bombardier CRJ-900 (regional jet)",
            "BCS3": "Airbus A220-300 (narrow-body)",
        }
        return aircraft_types.get(aircraft_type, f"{aircraft_type} (unknown type)")

    def _detect_flight_phase(self, altitude: float, speed: float) -> str:
        """Detect flight phase based on altitude and speed"""
        # Handle None values safely
        if altitude is None:
            altitude = 0
        if speed is None:
            speed = 0

        # Convert to float if needed
        try:
            altitude = float(altitude) if altitude != "ground" else 0
            speed = float(speed)
        except (ValueError, TypeError):
            return "unknown_phase"

        if altitude == 0 or altitude == "ground":
            if speed < 3:
                return "parked/stationary"
            elif speed < 30:
                return "taxiing"
            else:
                return "takeoff_roll"
        elif altitude < 500:
            if speed > 100:
                return "takeoff_climb"
            else:
                return "approach_final"
        elif altitude < 3000:
            if speed > 200:
                return "departure_climb"
            else:
                return "approach_descent"
        else:
            return "en_route"

    def _calculate_runway_proximity(self, lat: float, lon: float) -> str:
        """Calculate proximity to JFK runways"""
        # Handle None values
        if lat is None or lon is None:
            return "unknown_location"

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return "invalid_coordinates"

        # JFK runway coordinates (approximate thresholds)
        jfk_runways = {
            "04L/22R": {"lat": 40.6413, "lon": -73.7781},
            "04R/22L": {"lat": 40.6295, "lon": -73.7624},
            "08L/26R": {"lat": 40.6518, "lon": -73.7858},
            "08R/26L": {"lat": 40.6476, "lon": -73.7624},
            "13L/31R": {"lat": 40.6200, "lon": -73.7900},
            "13R/31L": {"lat": 40.6500, "lon": -73.7700},
        }

        closest_runway = None
        min_distance = float("inf")

        for runway, coords in jfk_runways.items():
            # Simple distance calculation
            distance = ((lat - coords["lat"]) ** 2 + (lon - coords["lon"]) ** 2) ** 0.5
            if distance < min_distance:
                min_distance = distance
                closest_runway = runway

        # Convert to approximate distance in feet (very rough)
        distance_feet = min_distance * 364000  # Rough conversion

        if distance_feet < 500:
            return f"on/near {closest_runway}"
        elif distance_feet < 2000:
            return f"close to {closest_runway}"
        else:
            return f"distant from runways"

    def _determine_airport_area(self, lat: float, lon: float) -> str:
        """Determine which area of JFK the aircraft is in"""
        # Handle None values
        if lat is None or lon is None:
            return "unknown_area"

        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return "invalid_coordinates"

        # JFK terminal areas (approximate)
        jfk_lat = 40.6413
        jfk_lon = -73.7781

        # Relative to JFK center
        lat_offset = lat - jfk_lat
        lon_offset = lon - jfk_lon

        if abs(lat_offset) < 0.005 and abs(lon_offset) < 0.005:
            return "terminal_area"
        elif lat_offset > 0:
            return "north_field"
        else:
            return "south_field"

    def _get_empty_state(self, error: Optional[str] = None) -> Dict[str, Any]:
        """Return empty aircraft state structure"""
        state = {
            "all_aircraft": [],
            "jfk_ground_aircraft": [],
            "jfk_air_aircraft": [],
            "runway_occupancy": {},
            "total_aircraft_count": 0,
            "timestamp": datetime.now().isoformat(),
        }
        if error:
            state["error"] = error
        return state

    def _calculate_runway_occupancy(
        self, ground_aircraft: Dict
    ) -> Dict[str, Optional[str]]:
        """Calculate which aircraft are occupying which runways (basic implementation)"""
        # TODO: Implement sophisticated runway occupancy detection
        # This would require:
        # - Precise aircraft position analysis
        # - JFK runway coordinate mapping
        # - Speed/movement pattern analysis

        runway_occupancy = {
            "04L/22R": None,
            "04R/22L": None,
            "13L/31R": None,
            "13R/31L": None,
        }

        # Basic implementation - could be enhanced with position analysis
        for callsign, aircraft in ground_aircraft.items():
            speed = aircraft.get("speed", 0)
            # If aircraft is moving fast on ground, might be on runway
            if isinstance(speed, (int, float)) and speed > 50:  # kts
                # This is a simplified heuristic - needs improvement
                runway_occupancy["13R/31L"] = callsign  # Example assignment
                break

        return runway_occupancy


class ValidationDatasetBuilder:
    """Build and manage validation dataset files"""

    def __init__(self, records_per_file: int = 100):
        self.records_per_file = records_per_file
        self.current_batch: List[Dict[str, Any]] = []
        self.file_counter = 1
        self.total_records = 0

    def add_record(self, record: ValidationRecord):
        """Add a validation record to the current batch"""
        self.current_batch.append(asdict(record))
        self.total_records += 1

        print(
            f"📝 Added validation record {self.total_records}: {record.atc_command['command_type']}"
        )

        # Check if batch is full
        if len(self.current_batch) >= self.records_per_file:
            self._save_batch()

    def _save_batch(self):
        """Save current batch to JSON file in validation-dataset folder"""
        if not self.current_batch:
            return

        # Create validation-dataset directory if it doesn't exist
        dataset_dir = os.path.join(os.path.dirname(__file__), "validation-dataset")
        os.makedirs(dataset_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"t{self.file_counter}-{timestamp}.json"
        filepath = os.path.join(dataset_dir, filename)

        try:
            with open(filepath, "w") as f:
                json.dump(self.current_batch, f, indent=2)

            print(
                f"💾 Saved batch {self.file_counter}: {len(self.current_batch)} records to validation-dataset/{filename}"
            )

            # Reset for next batch
            self.current_batch = []
            self.file_counter += 1

        except Exception as e:
            print(f"❌ Error saving batch: {e}")

    def finalize(self):
        """Save any remaining records in the current batch"""
        if self.current_batch:
            self._save_batch()
        print(f"✅ Dataset complete: {self.total_records} total validation records")
