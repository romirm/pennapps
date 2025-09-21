#!/usr/bin/env python3.9
"""
Demo script to analyze validation dataset and show test cases with non-unknown command types

This script loads all JSON files from the validation-dataset folder and analyzes them to:
1. Show overall statistics of command types
2. Filter and display test cases where command_type is not "unknown"
3. Provide detailed views of transcriptions, command analysis, and aircraft context

Usage Examples:
    python3.9 demo.py                              # Show first 10 non-unknown test cases
    python3.9 demo.py --stats-only                 # Show only statistics
    python3.9 demo.py --limit 20                   # Show first 20 test cases
    python3.9 demo.py --command-type taxi          # Show only taxi commands
    python3.9 demo.py --full --limit 3             # Show full explanations for 3 cases
"""

import json
import os
from typing import List, Dict, Any
from collections import Counter
import argparse


def load_validation_files(dataset_dir: str) -> List[Dict[str, Any]]:
    """Load all validation JSON files from the dataset directory"""
    all_records = []

    if not os.path.exists(dataset_dir):
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return all_records

    json_files = [f for f in os.listdir(dataset_dir) if f.endswith(".json")]

    if not json_files:
        print(f"❌ No JSON files found in {dataset_dir}")
        return all_records

    print(f"📂 Found {len(json_files)} validation files:")

    for filename in sorted(json_files):
        filepath = os.path.join(dataset_dir, filename)
        try:
            with open(filepath, "r") as f:
                records = json.load(f)
                all_records.extend(records)
                print(f"  ✅ {filename}: {len(records)} records")
        except Exception as e:
            print(f"  ❌ Error loading {filename}: {e}")

    print(f"\n📊 Total records loaded: {len(all_records)}")
    return all_records


def analyze_command_types(records: List[Dict[str, Any]]) -> Dict[str, int]:
    """Analyze distribution of command types"""
    command_types = []

    for record in records:
        atc_command = record.get("atc_command", {})
        command_type = atc_command.get("command_type", "missing")
        command_types.append(command_type)

    return Counter(command_types)


def filter_non_unknown_commands(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter records where command_type is not 'unknown'"""
    filtered_records = []

    for record in records:
        atc_command = record.get("atc_command", {})
        command_type = atc_command.get("command_type", "missing")

        if command_type.lower() not in ["unknown", "missing", "acknowledgment", ""]:
            filtered_records.append(record)

    return filtered_records


def display_test_case(record: Dict[str, Any], index: int, show_full: bool = False):
    """Display a single test case in a formatted way"""
    atc_command = record.get("atc_command", {})

    print(f"\n{'='*80}")
    print(f"🎯 TEST CASE #{index + 1}")
    print(f"{'='*80}")

    # Basic info
    print(f"📅 Timestamp: {record.get('timestamp_speech_start', 'N/A')}")
    print(f"🆔 Record ID: {record.get('record_id', 'N/A')}")
    print(f"⏱️  Processing Lag: {record.get('processing_lag_seconds', 'N/A'):.2f}s")

    # Command info
    command_type = atc_command.get("command_type", "N/A")
    confidence = atc_command.get("confidence_score", "N/A")

    print(f"\n🎙️  TRANSCRIPTION:")
    print(f"   \"{atc_command.get('raw_transcription', 'N/A')}\"")

    print(f"\n📋 COMMAND ANALYSIS:")
    print(f"   Type: {command_type.upper()}")
    print(f"   Confidence: {confidence}")

    affected_aircraft = atc_command.get("affected_aircraft", [])
    if affected_aircraft:
        print(f"   Affected Aircraft: {', '.join(affected_aircraft)}")

    # Extracted elements
    extracted_elements = atc_command.get("extracted_elements", {})
    if extracted_elements:
        print(f"\n🔍 EXTRACTED ELEMENTS:")
        for key, value in extracted_elements.items():
            if value:  # Only show non-empty values
                print(f"   {key.replace('_', ' ').title()}: {value}")

    # Aircraft context summary
    aircraft_states = record.get("aircraft_states", {})
    total_aircraft = aircraft_states.get("total_aircraft_count", 0)
    ground_aircraft = len(aircraft_states.get("jfk_ground_aircraft", []))
    air_aircraft = len(aircraft_states.get("jfk_air_aircraft", []))

    print(f"\n✈️  AIRCRAFT CONTEXT:")
    print(f"   Total JFK Aircraft: {total_aircraft}")
    print(f"   Ground: {ground_aircraft}, Air: {air_aircraft}")

    if show_full:
        # Show full explanation if requested
        explanation = atc_command.get("processed_explanation", "")
        if explanation:
            print(f"\n💡 FULL EXPLANATION:")
            # Truncate very long explanations
            if len(explanation) > 1000:
                print(f"   {explanation[:1000]}...")
            else:
                print(f"   {explanation}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze validation dataset for non-unknown command types"
    )
    parser.add_argument(
        "--full", action="store_true", help="Show full explanations for each test case"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of test cases to display (default: 10)",
    )
    parser.add_argument(
        "--command-type", type=str, help="Filter by specific command type"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Show only statistics, no individual test cases",
    )

    args = parser.parse_args()

    # Load validation data
    dataset_dir = os.path.join(os.path.dirname(__file__), "validation-dataset")
    records = load_validation_files(dataset_dir)

    if not records:
        return

    # Analyze command types
    print(f"\n📈 COMMAND TYPE DISTRIBUTION:")
    command_type_counts = analyze_command_types(records)

    for command_type, count in command_type_counts.most_common():
        percentage = (count / len(records)) * 100
        print(f"   {command_type:<15}: {count:>4} ({percentage:>5.1f}%)")

    # Filter non-unknown commands
    filtered_records = filter_non_unknown_commands(records)

    print(f"\n🎯 NON-UNKNOWN COMMANDS:")
    print(f"   Total Records: {len(records)}")
    print(f"   Non-Unknown: {len(filtered_records)}")
    print(f"   Percentage: {(len(filtered_records) / len(records)) * 100:.1f}%")

    if args.stats_only:
        return

    # Apply command type filter if specified
    if args.command_type:
        filtered_records = [
            r
            for r in filtered_records
            if r.get("atc_command", {}).get("command_type", "").lower()
            == args.command_type.lower()
        ]
        print(
            f"\n🔍 Filtered for command type '{args.command_type}': {len(filtered_records)} records"
        )

    if not filtered_records:
        print("\n❌ No matching records found!")
        return

    # Display test cases
    display_limit = min(args.limit, len(filtered_records))

    print(f"\n🚀 DISPLAYING {display_limit} TEST CASES:")

    for i in range(display_limit):
        display_test_case(filtered_records[i], i, show_full=args.full)

    if len(filtered_records) > display_limit:
        print(f"\n... and {len(filtered_records) - display_limit} more records")
        print(f"\nUse --limit {len(filtered_records)} to see all records")


if __name__ == "__main__":
    main()
