#!/usr/bin/env python3.9
"""
Demonstration of Enhanced Fragment Detection System

This script shows how the enhanced IFA transcription system now connects
fragmented ATC communications across multiple transcription segments.

Example: "Air France" → "9 7 0" → "taxi via alpha"
Now recognized as: "Air France 970, taxi via alpha"
"""

import asyncio
from datetime import datetime
from ifa_transcriber import JFKContextualTranscriber
from ifa_components import AircraftStateManager


async def demonstrate_fragment_reconstruction():
    print("🎯 ENHANCED FRAGMENT DETECTION DEMONSTRATION")
    print("=" * 60)
    print()

    # Initialize system
    transcriber = JFKContextualTranscriber()
    aircraft_manager = AircraftStateManager()
    aircraft_state = await aircraft_manager.get_current_aircraft_state()

    print(
        f"📡 Live aircraft context: {aircraft_state.get('total_aircraft_count', 0)} aircraft at JFK"
    )
    print()

    # Demonstration scenarios
    scenarios = [
        {
            "name": "Fragmented Taxi Clearance",
            "fragments": [
                "Air France",
                "9 7 0",
                "taxi via alpha kilo hold short runway 13R",
            ],
            "description": "Controller instruction split across radio breaks",
        },
        {
            "name": "Split Frequency Change",
            "fragments": ["Delta 671", "contact ground", "1 2 1 point 9"],
            "description": "Frequency handoff with number fragments",
        },
        {
            "name": "Pilot Readback Chain",
            "fragments": [
                "JetBlue 102 taxi alpha",
                "roger",
                "alpha kilo hold short 13R",
            ],
            "description": "Pilot confirming and completing instruction",
        },
    ]

    for scenario_num, scenario in enumerate(scenarios, 1):
        print(f"🔬 SCENARIO {scenario_num}: {scenario['name']}")
        print(f"   {scenario['description']}")
        print()

        for i, fragment in enumerate(scenario["fragments"], 1):
            print(f'   Fragment {i}: "{fragment}"')

            # Get AI analysis
            explanation = transcriber.explain_atc_communication(
                fragment, aircraft_state
            )

            if explanation and not explanation.startswith("⚠️"):
                # Extract key parts of the analysis
                lines = explanation.split("\n")
                fragment_analysis = None
                reconstructed = None
                callsigns = None
                command_type = None

                for line in lines:
                    if line.startswith("Fragment Analysis:"):
                        fragment_analysis = line[18:].strip()[:150] + "..."
                    elif line.startswith("Reconstructed Communication:"):
                        reconstructed = line[28:].strip()
                    elif line.startswith("Callsigns:"):
                        callsigns = line[10:].strip()
                    elif line.startswith("Command Type:"):
                        command_type = line[13:].strip()

                print(f"   🔍 Analysis: {fragment_analysis}")
                if (
                    reconstructed
                    and reconstructed != "N/A"
                    and "no reconstruction" not in reconstructed.lower()
                ):
                    print(f"   🔗 Reconstructed: {reconstructed}")
                if callsigns:
                    print(f"   ✈️  Callsigns: {callsigns}")
                if command_type:
                    print(f"   📋 Command: {command_type}")

                # Add to conversation history (simulating real system)
                fragment_type = transcriber._analyze_fragment_type(fragment)
                history_entry = {
                    "transcription": fragment,
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "fragment_type": fragment_type,
                    "explanation_preview": explanation[:100] + "...",
                }
                transcriber.conversation_history.append(history_entry)

            print()

        print("-" * 60)
        print()

    # Show conversation history summary
    print("📚 CONVERSATION HISTORY ANALYSIS:")
    print(f"   Stored fragments: {len(transcriber.conversation_history)}")

    fragment_types = {}
    for entry in transcriber.conversation_history:
        ftype = entry["fragment_type"]
        fragment_types[ftype] = fragment_types.get(ftype, 0) + 1

    print("   Fragment type distribution:")
    for ftype, count in fragment_types.items():
        print(f"     • {ftype}: {count}")

    print()
    print("🎯 KEY IMPROVEMENTS:")
    print("   ✅ Fragments now connected across transcription segments")
    print("   ✅ Orphaned callsigns linked to subsequent instructions")
    print("   ✅ Flight numbers connected to airline names")
    print("   ✅ Partial instructions completed from context")
    print("   ✅ Enhanced conversation continuity for AI agent training")


if __name__ == "__main__":
    asyncio.run(demonstrate_fragment_reconstruction())
