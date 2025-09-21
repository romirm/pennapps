#!/usr/bin/env python3
"""
Informed Fast ATC Transcriber (IFA) - File-Based Version

Combines live ATC transcription with real-time aircraft position data and saves
ATC radio chatter interpretations to a text file for advisor consumption.
Focuses on JFK airport operations.
"""

import asyncio
import json
import time
import uuid
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import queue
import re

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastatc_transcriber import FastATCTranscriber
    from client import PlaneMonitor
    from ifa_components import (
        ATCCommandParser,
        AircraftStateManager,
    )
    import requests
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(
        "Please ensure fastatc_transcriber.py, client.py, and ifa_components.py are available"
    )
    sys.exit(1)


class JFKContextualTranscriber(FastATCTranscriber):
    """Enhanced FastATCTranscriber with JFK-specific context and knowledge"""

    def explain_atc_communication(
        self, transcription: str, aircraft_context: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """Enhanced explanation with comprehensive JFK context and real-time aircraft data"""
        if not self.cerebras_api_key:
            return "⚠️  Cerebras API key not configured"

        try:
            headers = {
                "Authorization": f"Bearer {self.cerebras_api_key}",
                "Content-Type": "application/json",
            }

            # Build enhanced context from recent conversation history
            context_section = ""
            if self.conversation_history:
                context_section = self._build_enhanced_conversation_context()
                context_section += "\n"

            # Build aircraft context section
            aircraft_context_section = ""
            if aircraft_context:
                aircraft_context_section = self._build_aircraft_context_prompt(
                    aircraft_context
                )

            # Enhanced JFK-specific prompt with fragment detection
            prompt = f"""You are an expert ATC interpreter for JOHN F. KENNEDY INTERNATIONAL AIRPORT (KJFK) operations analyzing transcribed audio.

ENHANCED JFK CONTEXT:
- Four runways: 04L/22R (14,511ft), 04R/22L (8,400ft), 13L/31R (10,000ft), 13R/31L (14,511ft)
- Ground Control: 121.9 (North), 121.65 (South)
- Tower: 119.1 (04R/22L, 13L/31R), 123.9 (04L/22R, 13R/31L)
- Major taxiways: A, B, C, D, K (Kilo), complex hot spots
- Terminals: T1 (1-11), T4 (A1-A8, B20-B48), T5 (1-30 JetBlue), T7 (1-12), T8 (1-59 American)

{aircraft_context_section}

CRITICAL: FRAGMENT DETECTION & CONVERSATION CONTINUITY
⚠️ ATC communications are often fragmented across multiple transcription segments due to:
- Radio transmission breaks, static, overlapping voices
- Pilot readbacks split from controller instructions
- Callsigns separated from instructions (e.g., "Air France" → "9 7 0" → "taxi via alpha")

FRAGMENT ANALYSIS PRIORITIES:
1) **ALWAYS examine recent context** for incomplete communications that this segment might complete
2) **Look for orphaned callsigns** in recent history that need instructions
3) **Identify partial instructions** waiting for callsign completion
4) **Detect readback patterns** (pilot confirming controller instruction)
5) **Connect numbered sequences** (e.g., "9 7 0" likely completes "Air France")

FRAGMENT INDICATORS:
- Isolated callsigns without instructions ("Delta 671", "Air France")
- Isolated flight numbers ("9 7 0", "2 4 5")
- Partial instructions without callsigns ("taxi via alpha", "contact ground")
- Single words/phrases ("roger", "wilco", "affirm")
- Frequency fragments ("1 2 1 point 9")

Your job for AI agent training:
1) **FIRST**: Analyze if current transcription connects to recent fragments
2) Identify all aircraft callsigns and reconstruct complete communications
3) Extract runways, taxiways, frequencies, altitudes, speeds from combined context
4) Categorize command type considering fragment reconstruction
5) Assess operational significance of the complete communication chain

{context_section}

Output format:
Fragment Analysis: <Is this connected to recent communications? How?>
Reconstructed Communication: <Complete communication if fragments combined>
Callsigns: <list all aircraft mentioned, including from context>
Command Type: <category considering full context>
Extracted Elements: <runways, taxiways, frequencies, etc.>
Operational Significance: <what this accomplishes>
Confidence: <high/medium/low based on clarity and reconstruction>

Current Communication: "{transcription}"
"""

            data = {
                "model": "llama3.1-8b",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 4096,
                "temperature": 0.3,
            }

            response = requests.post(
                f"{self.cerebras_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code == 200:
                result = response.json()
                try:
                    if "choices" in result and len(result["choices"]) > 0:
                        choice = result["choices"][0]

                        if "message" in choice and "content" in choice["message"]:
                            return choice["message"]["content"].strip()
                        elif "message" in choice and "reasoning" in choice["message"]:
                            return choice["message"]["reasoning"].strip()
                        elif "text" in choice:
                            return choice["text"].strip()

                    return "⚠️  Unexpected API response format"

                except KeyError as e:
                    return f"⚠️  API Response Error: {e}"
            else:
                return f"⚠️  API Error: {response.status_code}"

        except Exception as e:
            return f"⚠️  API Error: {str(e)}"

    def _build_aircraft_context_prompt(self, aircraft_context: Dict[str, Any]) -> str:
        """Build aircraft context section for the prompt"""
        context_lines = ["REAL-TIME AIRCRAFT CONTEXT:"]

        # Ground aircraft
        ground_aircraft = aircraft_context.get("jfk_ground_aircraft", [])
        if ground_aircraft:
            context_lines.append(
                f"\nGROUND AIRCRAFT ({len(ground_aircraft)} aircraft):"
            )
            for aircraft in ground_aircraft[:10]:  # Limit to first 10 for prompt size
                callsign = aircraft.get("callsign", "Unknown")
                aircraft_type = aircraft.get(
                    "aircraft_type_description",
                    aircraft.get("aircraft_type", "Unknown"),
                )
                flight_phase = aircraft.get("flight_phase", "unknown")
                runway_proximity = aircraft.get("runway_proximity", "unknown location")
                airport_area = aircraft.get("airport_area", "unknown area")
                speed = aircraft.get("speed", "N/A")

                context_lines.append(
                    f"  • {callsign}: {aircraft_type}, {flight_phase}, {runway_proximity}, {airport_area} (speed: {speed} kts)"
                )

        # Air aircraft
        air_aircraft = aircraft_context.get("jfk_air_aircraft", [])
        if air_aircraft:
            context_lines.append(f"\nAIR AIRCRAFT ({len(air_aircraft)} aircraft):")
            for aircraft in air_aircraft[:5]:  # Limit to first 5 for prompt size
                callsign = aircraft.get("callsign", "Unknown")
                aircraft_type = aircraft.get(
                    "aircraft_type_description",
                    aircraft.get("aircraft_type", "Unknown"),
                )
                flight_phase = aircraft.get("flight_phase", "unknown")
                altitude = aircraft.get("altitude", "N/A")
                speed = aircraft.get("speed", "N/A")

                context_lines.append(
                    f"  • {callsign}: {aircraft_type}, {flight_phase}, {altitude} ft, {speed} kts"
                )

        # Runway occupancy
        runway_occupancy = aircraft_context.get("runway_occupancy", {})
        if runway_occupancy:
            context_lines.append(f"\nRUNWAY OCCUPANCY:")
            for runway, count in runway_occupancy.items():
                if count is not None and isinstance(count, (int, float)) and count > 0:
                    context_lines.append(f"  • {runway}: {count} aircraft")
                elif count is not None and not isinstance(count, (int, float)):
                    # Handle string callsigns (like "DAL671")
                    context_lines.append(f"  • {runway}: {count}")

        context_lines.append(
            f"\nTotal JFK area aircraft: {aircraft_context.get('total_aircraft_count', 0)}"
        )
        context_lines.append("")  # Empty line for formatting

        return "\n".join(context_lines)

    def _build_enhanced_conversation_context(self) -> str:
        """Build enhanced conversation context with fragment detection"""
        if not self.conversation_history:
            return ""

        context_lines = [
            "RECENT ATC CONVERSATION HISTORY (analyze for fragments & connections):",
            "=" * 70,
        ]

        # Get recent history with timestamps if available
        recent_history = self.conversation_history[
            -8:
        ]  # Increased from 5 to 8 for better context

        for i, entry in enumerate(recent_history, 1):
            # Handle both simple strings and enhanced entries
            if isinstance(entry, dict):
                transcription = entry.get("transcription", "")
                timestamp = entry.get("timestamp", "")
                fragment_type = entry.get("fragment_type", "unknown")
                context_lines.append(
                    f'[{i}] {timestamp} ({fragment_type}): "{transcription}"'
                )
            else:
                # Legacy string format
                fragment_analysis = self._analyze_fragment_type(entry)
                context_lines.append(f'[{i}] ({fragment_analysis}): "{entry}"')

        # Add fragment reconstruction hints
        context_lines.extend(
            [
                "",
                "FRAGMENT ANALYSIS HINTS:",
                "• Look for orphaned callsigns that need completion",
                "• Connect flight numbers to airline names (e.g., Air France + 970)",
                "• Link partial instructions to recent callsigns",
                "• Identify controller→pilot vs pilot→controller patterns",
                "• Watch for readback confirmations of previous instructions",
                "",
            ]
        )

        return "\n".join(context_lines)

    def _analyze_fragment_type(self, transcription: str) -> str:
        """Analyze what type of fragment this transcription represents"""
        text = transcription.lower().strip()

        # Common fragment patterns
        if len(text.split()) <= 2:
            if any(
                airline in text
                for airline in [
                    "delta",
                    "american",
                    "jetblue",
                    "air france",
                    "united",
                    "southwest",
                ]
            ):
                return "ORPHANED_CALLSIGN"
            elif text.replace(" ", "").isdigit() or (
                "point" in text and any(c.isdigit() for c in text)
            ):
                return "FLIGHT_NUMBER/FREQ"
            elif text in ["roger", "wilco", "affirm", "negative", "standby"]:
                return "ACKNOWLEDGMENT"
            else:
                return "FRAGMENT"

        # Check for common instruction patterns without callsigns
        instruction_keywords = [
            "taxi",
            "contact",
            "hold",
            "cleared",
            "turn",
            "climb",
            "descend",
        ]
        if any(keyword in text for keyword in instruction_keywords) and not any(
            airline in text for airline in ["delta", "american", "jetblue"]
        ):
            return "ORPHANED_INSTRUCTION"

        # Check for complete communications
        has_callsign = any(
            airline in text
            for airline in ["delta", "american", "jetblue", "air france", "united"]
        )
        has_instruction = any(keyword in text for keyword in instruction_keywords)

        if has_callsign and has_instruction:
            return "COMPLETE_COMM"
        elif has_callsign:
            return "CALLSIGN_ONLY"
        elif has_instruction:
            return "INSTRUCTION_ONLY"
        else:
            return "UNCLEAR"


class InformedATCTranscriber:
    """File-based ATC transcriber that saves interpretations to transcribed_chatter.txt"""

    def __init__(self):
        self.transcriber = JFKContextualTranscriber()
        self.aircraft_manager = AircraftStateManager()
        self.command_parser = ATCCommandParser()

        # File-based storage setup
        self.chatter_file = os.path.join(
            os.path.dirname(__file__), "transcribed_chatter.txt"
        )

        # Initialize the file with header
        self._initialize_chatter_file()

        # Speech processing tracking
        self.speech_segments = {}

        print("🚀 File-Based ATC Transcriber initialized for JFK operations")
        print("📄 Saving ATC interpretations to transcribed_chatter.txt")
        print(f"📁 File location: {self.chatter_file}")

    def _initialize_chatter_file(self):
        """Initialize the chatter file with header information"""
        header = f"""# JFK ATC Radio Chatter Interpretations
# Generated by IFA Transcriber File-Based (ifa_transcriber_fb.py)
# Started: {datetime.now().isoformat()}
# Format: [TIMESTAMP] TRANSCRIPTION -> INTERPRETATION
# 
"""
        with open(self.chatter_file, "w") as f:
            f.write(header)

    def _write_chatter_to_file(
        self, transcription: str, interpretation: str, aircraft_mentioned: list = None
    ):
        """Write ATC chatter interpretation to file"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            aircraft_str = (
                f" [{', '.join(aircraft_mentioned)}]" if aircraft_mentioned else ""
            )

            entry = (
                f"[{timestamp}]{aircraft_str} {transcription}\n-> {interpretation}\n\n"
            )

            with open(self.chatter_file, "a") as f:
                f.write(entry)

        except Exception as e:
            print(f"❌ Error writing to chatter file: {e}")

    def process_audio_queue(self):
        """Enhanced audio processing with aircraft state correlation"""
        print("🔄 Started informed audio processing thread...")

        while self.transcriber.is_recording or not self.transcriber.audio_queue.empty():
            try:
                # Get audio data from queue
                audio_data = self.transcriber.audio_queue.get(timeout=1)

                # Track speech segment timing
                speech_id = str(uuid.uuid4())
                speech_start_time = time.time()
                self.speech_segments[speech_id] = speech_start_time

                # Update progress tracking
                self.transcriber.chunks_processed += 1
                queue_size = self.transcriber.audio_queue.qsize()
                print(
                    f"🎵 Processing speech segment #{self.transcriber.chunks_processed}... ({queue_size} remaining)"
                )

                # Save and transcribe audio
                audio_file = self.transcriber.save_audio_chunk(audio_data)
                transcription = self.transcriber.transcribe_audio(audio_file)

                if transcription and len(transcription.strip()) > 5:
                    print(f"\n📝 ATC Transcription: {transcription}")

                    # Fetch aircraft state at transcription completion (simplified timing approach)
                    print("🛩️ Fetching correlated aircraft state...")
                    aircraft_state = asyncio.run(
                        self.aircraft_manager.get_current_aircraft_state()
                    )

                    # Get enhanced explanation with aircraft context
                    explanation = self.transcriber.explain_atc_communication(
                        transcription, aircraft_state
                    )

                    if explanation:
                        print(f"💡 Enhanced JFK Analysis: {explanation}")

                        # DEBUG: Print aircraft names being passed to transcription interpreter
                        ground_aircraft = aircraft_state.get("jfk_ground_aircraft", [])
                        air_aircraft = aircraft_state.get("jfk_air_aircraft", [])

                        print(
                            f"🔍 DEBUG: Current aircraft being passed to interpreter:"
                        )

                        # Fix: Aircraft data structure uses flight number as key, not callsign field
                        if ground_aircraft:
                            # Check if ground_aircraft is a list of dicts or dict values
                            if isinstance(ground_aircraft, list):
                                ground_callsigns = [
                                    ac.get("callsign", "N/A")
                                    for ac in ground_aircraft
                                    if isinstance(ac, dict)
                                ]
                            else:
                                # If it's a dict, the keys are the callsigns
                                ground_callsigns = list(ground_aircraft.keys())

                            print(
                                f"   📍 Ground aircraft ({len(ground_callsigns)}): {', '.join(ground_callsigns)}"
                            )
                        else:
                            print(f"   📍 Ground aircraft: None")

                        if air_aircraft:
                            # Check if air_aircraft is a list of dicts or dict values
                            if isinstance(air_aircraft, list):
                                air_callsigns = [
                                    ac.get("callsign", "N/A")
                                    for ac in air_aircraft
                                    if isinstance(ac, dict)
                                ]
                            else:
                                # If it's a dict, the keys are the callsigns
                                air_callsigns = list(air_aircraft.keys())

                            print(
                                f"   ✈️  Air aircraft ({len(air_callsigns)}): {', '.join(air_callsigns)}"
                            )
                        else:
                            print(f"   ✈️  Air aircraft: None")

                        total_aircraft = aircraft_state.get("total_aircraft_count", 0)
                        print(f"   📊 Total aircraft in JFK area: {total_aircraft}")

                        if ground_aircraft and len(ground_aircraft) > 0:
                            first_ground = (
                                ground_aircraft[0]
                                if isinstance(ground_aircraft, list)
                                else list(ground_aircraft.values())[0]
                            )

                        # Parse command for structured data
                        parsed_command = self.command_parser.parse_command(
                            transcription, explanation
                        )

                        # Extract aircraft mentioned
                        aircraft_mentioned = parsed_command.get("affected_aircraft", [])

                        # Write to file instead of creating validation records
                        self._write_chatter_to_file(
                            transcription, explanation, aircraft_mentioned
                        )
                        print(f"💾 Saved to transcribed_chatter.txt")

                        # Add enhanced entry to conversation history for context
                        fragment_type = self.transcriber._analyze_fragment_type(
                            transcription
                        )
                        history_entry = {
                            "transcription": transcription,
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "fragment_type": fragment_type,
                            "explanation_preview": (
                                explanation[:100] + "..."
                                if len(explanation) > 100
                                else explanation
                            ),
                        }

                        self.transcriber.conversation_history.append(history_entry)
                        if (
                            len(self.transcriber.conversation_history)
                            > self.transcriber.max_history_items
                        ):
                            self.transcriber.conversation_history.pop(0)

                    print("-" * 80)
                else:
                    # Check if we should ignore this (pure acknowledgments, etc.)
                    if transcription and len(transcription.strip()) > 0:
                        ignored_types = ["roger", "copy", "wilco", "affirmative"]
                        if any(
                            ignored in transcription.lower()
                            for ignored in ignored_types
                        ):
                            print(f"🚫 Ignored pure acknowledgment: '{transcription}'")
                        else:
                            print(
                                f"📻 Short transmission (not processed): '{transcription}'"
                            )
                    else:
                        print("📻 No clear ATC transmission detected in audio chunk")

                # Clean up speech segment tracking
                if speech_id in self.speech_segments:
                    del self.speech_segments[speech_id]

                # Mark task as done
                self.transcriber.audio_queue.task_done()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Processing error: {e}")
                try:
                    self.transcriber.audio_queue.task_done()
                except ValueError:
                    pass

        print("🔍 Exited informed processing loop")

    def run(self):
        """Run the informed ATC transcriber system"""
        print("🚀 Starting File-Based Fast ATC Transcriber for JFK...")
        print("📄 Saving ATC interpretations for advisor consumption")
        print("📡 Monitoring JFK ground aircraft and ATC communications")
        print()

        # Start enhanced audio processing thread
        processing_thread = threading.Thread(
            target=self.process_audio_queue, daemon=False
        )
        processing_thread.start()
        self.transcriber.processing_thread = processing_thread

        # Start recording (this will block until Ctrl+C)
        try:
            self.transcriber.start_recording()
        except KeyboardInterrupt:
            print("\n🛑 Stopping File-Based ATC Transcriber...")

        print("⏳ Finishing processing remaining audio...")

        # Wait for processing to complete
        timeout_seconds = 60
        start_time = time.time()

        while (
            not self.transcriber.audio_queue.empty()
            and (time.time() - start_time) < timeout_seconds
        ):
            current_size = self.transcriber.audio_queue.qsize()
            if current_size > 0:
                print(f"⏳ Still processing... {current_size} chunks remaining")
            time.sleep(2)

        if self.transcriber.audio_queue.empty():
            print("✅ All audio chunks processed successfully")
        else:
            remaining = self.transcriber.audio_queue.qsize()
            print(
                f"⚠️  Timeout reached. {remaining} chunks may not have been processed."
            )

        # Wait for processing thread to finish
        if (
            hasattr(self.transcriber, "processing_thread")
            and self.transcriber.processing_thread.is_alive()
        ):
            print("⏳ Waiting for processing thread to finish...")
            self.transcriber.processing_thread.join(timeout=10)

        print("✅ File-Based Fast ATC Transcriber stopped")
        print(f"📄 ATC interpretations saved to: {self.chatter_file}")


def main():
    """Main entry point"""
    print("File-Based Fast ATC Transcriber (IFA-FB)")
    print("JFK-Focused ATC Interpretation File Storage")
    print("=" * 60)

    # Check for required environment variables
    if not os.getenv("CEREBRAS_API_KEY"):
        print("\n⚠️  Setup Required:")
        print("1. Create a .env file in this directory")
        print("2. Add your Cerebras API key: CEREBRAS_API_KEY=your_key_here")
        print()

        response = input("Continue without Cerebras integration? (y/N): ").lower()
        if response != "y":
            return

    # Initialize and run the informed transcriber
    transcriber = InformedATCTranscriber()
    transcriber.run()


if __name__ == "__main__":
    main()
