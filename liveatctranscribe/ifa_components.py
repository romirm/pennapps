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

            return {
                "all_aircraft": list(ground_aircraft.values())
                + list(air_aircraft.values()),
                "jfk_ground_aircraft": list(ground_aircraft.values()),
                "jfk_air_aircraft": list(air_aircraft.values()),  # Currently within 5nm
                "runway_occupancy": runway_occupancy,
                "total_aircraft_count": aircraft_data.get("total_aircraft", 0),
                "timestamp": aircraft_data.get("timestamp", datetime.now().isoformat()),
            }

        except Exception as e:
            print(f"❌ Error fetching aircraft state: {e}")
            return self._get_empty_state(error=str(e))

    def _get_empty_state(self, error: str = None) -> Dict[str, Any]:
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
        self.current_batch = []
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
        """Save current batch to JSON file"""
        if not self.current_batch:
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"t{self.file_counter}-{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), filename)

        try:
            with open(filepath, "w") as f:
                json.dump(self.current_batch, f, indent=2)

            print(
                f"💾 Saved batch {self.file_counter}: {len(self.current_batch)} records to {filename}"
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
