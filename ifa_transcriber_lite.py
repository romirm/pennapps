#!/usr/bin/env python3
"""
IFA Transcriber Lite

Lightweight version of the Informed Fast ATC Transcriber designed to integrate
with the Flask app. Maintains in-memory ATC context for bottleneck prediction
enhancement.
"""

import asyncio
import json
import time
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import threading
import queue
import re
import requests
from collections import deque

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from liveatctranscribe.fastatc_transcriber import FastATCTranscriber
    from client import PlaneMonitor
    from liveatctranscribe.ifa_components import ATCCommandParser, AircraftStateManager
except ImportError as e:
    print(f"❌ Import error in IFA Lite: {e}")
    # Continue without transcription - can still provide mock context
    FastATCTranscriber = None
    ATCCommandParser = None
    AircraftStateManager = None


@dataclass
class ATCChatterEntry:
    """Single ATC communication entry for context"""

    timestamp: str
    transcription: str
    interpretation: str
    aircraft_mentioned: List[str]
    runways_mentioned: List[str]
    taxiways_mentioned: List[str]
    command_type: str  # taxi, clearance, hold, contact, etc.
    confidence: str  # high/medium/low
    fragment_type: str


class IFATranscriberLite:
    """Lightweight ATC transcriber for Flask app integration"""

    def __init__(self, max_context_entries: int = 50, context_window_minutes: int = 15):
        self.max_context_entries = max_context_entries
        self.context_window_minutes = context_window_minutes

        # In-memory context storage - thread-safe deque
        self.chatter_context = deque(maxlen=max_context_entries)
        self.context_lock = threading.RLock()

        # Initialize components if available
        self.transcriber = None
        self.command_parser = None
        self.aircraft_manager = None

        if FastATCTranscriber and ATCCommandParser and AircraftStateManager:
            try:
                self.transcriber = self._create_contextual_transcriber()
                self.command_parser = ATCCommandParser()
                self.aircraft_manager = AircraftStateManager()
                self.is_transcription_enabled = True
                print("🎧 IFA Transcriber Lite: Full transcription enabled")
            except Exception as e:
                print(f"⚠️  IFA Transcriber Lite: Transcription disabled ({e})")
                self.is_transcription_enabled = False
        else:
            self.is_transcription_enabled = False
            print("🎧 IFA Transcriber Lite: Mock mode (no transcription)")

        # Background processing
        self.is_running = False
        self.processing_thread = None

        print(
            f"✅ IFA Transcriber Lite initialized (context window: {context_window_minutes}min)"
        )

    def _create_contextual_transcriber(self):
        """Create JFK-contextual transcriber similar to full version"""

        class JFKContextualTranscriberLite(FastATCTranscriber):
            def explain_atc_communication(
                self,
                transcription: str,
                aircraft_context: Optional[Dict[str, Any]] = None,
            ) -> Optional[str]:
                if not self.cerebras_api_key:
                    return "⚠️ Cerebras API key not configured"

                try:
                    headers = {
                        "Authorization": f"Bearer {self.cerebras_api_key}",
                        "Content-Type": "application/json",
                    }

                    # Simplified JFK-specific prompt for lite version
                    prompt = f"""You are an expert ATC interpreter for JFK operations.

JFK CONTEXT:
- Runways: 04L/22R, 04R/22L, 13L/31R, 13R/31L
- Ground Control: 121.9 (North), 121.65 (South)  
- Tower: 119.1, 123.9
- Major taxiways: A, B, C, D, K

Analyze this ATC communication for bottleneck prediction:
"{transcription}"

Output format:
Aircraft: <callsigns mentioned>
Command Type: <taxi/clearance/hold/contact/other>
Runways: <runways mentioned>
Taxiways: <taxiways mentioned>
Operational Impact: <brief description>
Confidence: <high/medium/low>
"""

                    data = {
                        "model": "llama3.1-8b",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1024,
                        "temperature": 0.3,
                    }

                    response = requests.post(
                        f"{self.cerebras_base_url}/chat/completions",
                        headers=headers,
                        json=data,
                        timeout=15,
                    )

                    if response.status_code == 200:
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            choice = result["choices"][0]
                            if "message" in choice and "content" in choice["message"]:
                                return choice["message"]["content"].strip()

                    return f"⚠️ API Error: {response.status_code}"

                except Exception as e:
                    return f"⚠️ API Error: {str(e)}"

        return JFKContextualTranscriberLite()

    def process_audio_queue(self):
        """Live audio processing with airport state model building"""
        print("🔄 Started live ATC audio processing...")

        while self.transcriber.is_recording or not self.transcriber.audio_queue.empty():
            try:
                # Get audio data from queue
                audio_data = self.transcriber.audio_queue.get(timeout=1)

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

                    # Get enhanced explanation with aircraft context
                    explanation = None
                    if self.is_transcription_enabled and self.transcriber:
                        explanation = self.transcriber.explain_atc_communication(
                            transcription
                        )

                    if explanation:
                        print(f"💡 Enhanced JFK Analysis: {explanation}")

                    # Process transcription into airport state model
                    entry = self._process_transcription_to_state_model(
                        transcription, explanation
                    )

                    if entry:
                        self.add_chatter_entry(entry)
                        print(
                            f"🏛️ Updated airport state model: {entry.command_type} - {', '.join(entry.aircraft_mentioned)}"
                        )

                    print("-" * 80)
                else:
                    # Handle short/unclear transmissions
                    if transcription and len(transcription.strip()) > 0:
                        ignored_types = ["roger", "copy", "wilco", "affirmative"]
                        if any(
                            ignored in transcription.lower()
                            for ignored in ignored_types
                        ):
                            print(f"🚫 Ignored acknowledgment: '{transcription}'")
                        else:
                            print(f"📻 Short transmission: '{transcription}'")
                    else:
                        print("📻 No clear ATC transmission detected")

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

        print("🔍 Exited live ATC processing loop")

    def _process_transcription_to_state_model(
        self, transcription: str, explanation: str = None
    ) -> Optional[ATCChatterEntry]:
        """Process transcription into airport state model entry"""
        try:
            # Basic parsing for aircraft, runways, taxiways
            aircraft_mentioned = self._extract_aircraft_callsigns(transcription)
            runways_mentioned = self._extract_runways(transcription)
            taxiways_mentioned = self._extract_taxiways(transcription)
            command_type = self._classify_command_type(transcription)

            # Determine confidence based on explanation availability
            confidence = "high" if explanation and "⚠️" not in explanation else "medium"

            entry = ATCChatterEntry(
                timestamp=datetime.now().isoformat(),
                transcription=transcription,
                interpretation=explanation
                or "Basic parsing - no AI interpretation available",
                aircraft_mentioned=aircraft_mentioned,
                runways_mentioned=runways_mentioned,
                taxiways_mentioned=taxiways_mentioned,
                command_type=command_type,
                confidence=confidence,
                fragment_type="LIVE_ATC",
            )

            return entry

        except Exception as e:
            print(f"❌ Error processing transcription to state model: {e}")
            return None

    def start_live_transcription(self):
        """Start live ATC transcription and airport state modeling"""
        if not self.is_transcription_enabled:
            print("❌ Cannot start live transcription - audio components not available")
            print(
                "   Install required dependencies: pip install pyaudio faster-whisper"
            )
            return False

        if self.is_running:
            print("⚠️ Live transcription already running")
            return True

        print("🚀 Starting Live ATC Transcription for Airport State Modeling...")
        print("🏛️ Building real-time airport operations model")
        print("📡 Monitoring JFK ATC communications")
        print()

        # Start audio processing thread
        self.processing_thread = threading.Thread(
            target=self.process_audio_queue, daemon=False
        )
        self.processing_thread.start()
        self.transcriber.processing_thread = self.processing_thread

        # Start recording in a separate thread for Flask integration
        self.recording_thread = threading.Thread(
            target=self._start_recording_thread, daemon=False
        )
        self.recording_thread.start()
        
        self.is_running = True
        return True

    def _start_recording_thread(self):
        """Start recording in a separate thread"""
        try:
            self.transcriber.start_recording()
        except KeyboardInterrupt:
            print("\n🛑 Stopping Live ATC Transcription...")
            self.stop_live_transcription()
        except Exception as e:
            print(f"❌ Recording thread error: {e}")
            self.is_running = False

    def stop_live_transcription(self):
        """Stop live transcription"""
        if not self.is_running:
            return

        self.is_running = False

        if hasattr(self.transcriber, "stop_recording"):
            self.transcriber.stop_recording()

        print("⏳ Finishing processing remaining audio...")

        # Wait for processing to complete
        timeout_seconds = 30
        start_time = time.time()

        if hasattr(self.transcriber, "audio_queue"):
            while (
                not self.transcriber.audio_queue.empty()
                and (time.time() - start_time) < timeout_seconds
            ):
                current_size = self.transcriber.audio_queue.qsize()
                if current_size > 0:
                    print(f"⏳ Still processing... {current_size} chunks remaining")
                time.sleep(2)

        # Wait for threads to finish
        if hasattr(self, "recording_thread") and self.recording_thread.is_alive():
            print("⏳ Waiting for recording thread to finish...")
            self.recording_thread.join(timeout=5)
            
        if hasattr(self, "processing_thread") and self.processing_thread.is_alive():
            print("⏳ Waiting for processing thread to finish...")
            self.processing_thread.join(timeout=10)

        print("✅ Live ATC Transcription stopped")
        print(
            f"🏛️ Airport state model contains {len(self.chatter_context)} recent communications"
        )

    def add_chatter_entry(self, entry: ATCChatterEntry):
        """Add ATC chatter entry to in-memory context"""
        with self.context_lock:
            self.chatter_context.append(entry)

    def process_transcription(self, transcription: str) -> Optional[ATCChatterEntry]:
        """Process a single ATC transcription into context entry"""
        if not transcription or len(transcription.strip()) < 5:
            return None

        try:
            # Get interpretation if transcription is enabled
            interpretation = "Basic parsing - no AI interpretation available"
            confidence = "low"

            if self.is_transcription_enabled and self.transcriber:
                interpretation = self.transcriber.explain_atc_communication(
                    transcription
                )
                if interpretation and "⚠️" not in interpretation:
                    confidence = "high"
                else:
                    confidence = "medium"

            # Basic parsing for aircraft, runways, taxiways
            aircraft_mentioned = self._extract_aircraft_callsigns(transcription)
            runways_mentioned = self._extract_runways(transcription)
            taxiways_mentioned = self._extract_taxiways(transcription)
            command_type = self._classify_command_type(transcription)

            entry = ATCChatterEntry(
                timestamp=datetime.now().isoformat(),
                transcription=transcription,
                interpretation=interpretation or "No interpretation available",
                aircraft_mentioned=aircraft_mentioned,
                runways_mentioned=runways_mentioned,
                taxiways_mentioned=taxiways_mentioned,
                command_type=command_type,
                confidence=confidence,
                fragment_type="UNKNOWN",
            )

            self.add_chatter_entry(entry)
            return entry

        except Exception as e:
            print(f"❌ Error processing transcription: {e}")
            return None

    def _extract_aircraft_callsigns(self, text: str) -> List[str]:
        """Extract aircraft callsigns from text"""
        callsigns = []
        text_lower = text.lower()

        # Common airline patterns
        airline_patterns = [
            (r"delta\s*(\d+)", "DAL{}"),
            (r"american\s*(\d+)", "AAL{}"),
            (r"jetblue\s*(\d+)", "JBU{}"),
            (r"air\s*france\s*(\d+)", "AFR{}"),
            (r"united\s*(\d+)", "UAL{}"),
        ]

        for pattern, format_str in airline_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                callsigns.append(format_str.format(match))

        return callsigns

    def _extract_runways(self, text: str) -> List[str]:
        """Extract runway mentions from text"""
        runway_pattern = r"runway\s*(\d{1,2}[LRC]?)"
        matches = re.findall(runway_pattern, text.lower())
        return [match.upper() for match in matches]

    def _extract_taxiways(self, text: str) -> List[str]:
        """Extract taxiway mentions from text"""
        taxiway_pattern = r"(?:taxiway\s*|via\s*)([A-Z])\b"
        matches = re.findall(taxiway_pattern, text, re.IGNORECASE)
        return [match.upper() for match in matches]

    def _classify_command_type(self, text: str) -> str:
        """Classify the type of ATC command"""
        text_lower = text.lower()

        if any(word in text_lower for word in ["taxi", "via"]):
            return "taxi"
        elif any(word in text_lower for word in ["cleared", "takeoff", "departure"]):
            return "clearance"
        elif any(word in text_lower for word in ["hold", "wait"]):
            return "hold"
        elif any(word in text_lower for word in ["contact", "switch"]):
            return "contact"
        elif any(word in text_lower for word in ["turn", "heading"]):
            return "navigation"
        else:
            return "other"

    def get_current_context(self, minutes_back: int = None) -> Dict[str, Any]:
        """Get current ATC chatter context for bottleneck analysis"""
        if minutes_back is None:
            minutes_back = self.context_window_minutes

        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)

        with self.context_lock:
            # Filter entries within time window
            recent_entries = []
            for entry in reversed(self.chatter_context):  # Most recent first
                entry_time = datetime.fromisoformat(
                    entry.timestamp.replace("Z", "+00:00").replace("+00:00", "")
                )
                if entry_time >= cutoff_time:
                    recent_entries.append(entry)
                else:
                    break  # Entries are time-ordered

            # Build context summary
            context = {
                "time_window_minutes": minutes_back,
                "total_communications": len(recent_entries),
                "timestamp_generated": datetime.now().isoformat(),
                "recent_chatter": [],
                "active_aircraft": set(),
                "active_runways": set(),
                "active_taxiways": set(),
                "command_summary": {
                    "taxi": 0,
                    "clearance": 0,
                    "hold": 0,
                    "contact": 0,
                    "other": 0,
                },
                "high_confidence_count": 0,
            }

            # Process each entry
            for entry in recent_entries:
                # Add to recent chatter
                context["recent_chatter"].append(
                    {
                        "timestamp": entry.timestamp,
                        "transcription": entry.transcription,
                        "interpretation": entry.interpretation,
                        "aircraft": entry.aircraft_mentioned,
                        "command_type": entry.command_type,
                        "confidence": entry.confidence,
                    }
                )

                # Aggregate data
                context["active_aircraft"].update(entry.aircraft_mentioned)
                context["active_runways"].update(entry.runways_mentioned)
                context["active_taxiways"].update(entry.taxiways_mentioned)
                context["command_summary"][entry.command_type] += 1

                if entry.confidence == "high":
                    context["high_confidence_count"] += 1

            # Convert sets to lists for JSON serialization
            context["active_aircraft"] = list(context["active_aircraft"])
            context["active_runways"] = list(context["active_runways"])
            context["active_taxiways"] = list(context["active_taxiways"])

            return context

    def get_aircraft_specific_context(
        self, aircraft_callsigns: List[str], minutes_back: int = None
    ) -> List[Dict[str, Any]]:
        """Get context entries specifically mentioning given aircraft"""
        if minutes_back is None:
            minutes_back = self.context_window_minutes

        cutoff_time = datetime.now() - timedelta(minutes=minutes_back)

        with self.context_lock:
            relevant_entries = []
            for entry in reversed(self.chatter_context):
                entry_time = datetime.fromisoformat(
                    entry.timestamp.replace("Z", "+00:00").replace("+00:00", "")
                )
                if entry_time < cutoff_time:
                    break

                # Check if any target aircraft are mentioned
                if any(
                    aircraft in entry.aircraft_mentioned
                    for aircraft in aircraft_callsigns
                ):
                    relevant_entries.append(
                        {
                            "timestamp": entry.timestamp,
                            "transcription": entry.transcription,
                            "interpretation": entry.interpretation,
                            "aircraft_mentioned": entry.aircraft_mentioned,
                            "command_type": entry.command_type,
                            "confidence": entry.confidence,
                        }
                    )

            return relevant_entries

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about current context"""
        with self.context_lock:
            if not self.chatter_context:
                return {
                    "total_entries": 0,
                    "status": "No data",
                    "transcription_enabled": self.is_transcription_enabled,
                    "background_processing": self.is_running,
                    "context_window_minutes": self.context_window_minutes,
                }

            oldest_entry = self.chatter_context[0]
            newest_entry = self.chatter_context[-1]

            return {
                "total_entries": len(self.chatter_context),
                "oldest_timestamp": oldest_entry.timestamp,
                "newest_timestamp": newest_entry.timestamp,
                "transcription_enabled": self.is_transcription_enabled,
                "background_processing": self.is_running,
                "context_window_minutes": self.context_window_minutes,
            }


# Global instance for Flask app integration
_ifa_lite_instance = None


def get_ifa_lite() -> IFATranscriberLite:
    """Get global IFA Transcriber Lite instance"""
    global _ifa_lite_instance
    if _ifa_lite_instance is None:
        _ifa_lite_instance = IFATranscriberLite()
    return _ifa_lite_instance


if __name__ == "__main__":
    # Test the lite transcriber
    print("IFA Transcriber Lite - Live ATC Airport State Modeling")
    print("=" * 60)

    ifa = IFATranscriberLite()

    if ifa.is_transcription_enabled:
        print("🎧 Full transcription capabilities available")
        print("🚀 Starting live ATC transcription...")
        print("   Press Ctrl+C to stop")
        print()

        # Start live transcription (this will block)
        ifa.start_live_transcription()
    else:
        print("📻 Running in test mode with sample data...")

        # Test adding some chatter
        test_transcriptions = [
            "Delta 671 taxi via alpha to runway 22L",
            "JetBlue 479 contact ground 121.9",
            "American 374 hold short runway 13R",
        ]

        for transcription in test_transcriptions:
            entry = ifa.process_transcription(transcription)
            if entry:
                print(f"✅ Processed: {entry.transcription}")

        # Test getting context
        context = ifa.get_current_context()
        print(f"\n📊 Airport State Model Summary:")
        print(f"  - Communications: {context['total_communications']}")
        print(f"  - Active Aircraft: {context['active_aircraft']}")
        print(f"  - Active Runways: {context['active_runways']}")
        print(f"  - Command Summary: {context['command_summary']}")

        print("\n✅ IFA Transcriber Lite test completed")
        print("\n💡 To enable live transcription:")
        print("   pip install pyaudio faster-whisper")
        print("   Then run this script again")
