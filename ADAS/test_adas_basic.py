#!/usr/bin/env python3
"""
Basic ADAS System Test
Quick validation that the system components work correctly
"""

import asyncio
import json
import sys
import os

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adas_system import ADASystem
from task_agent import TaskAgent
from meta_agent import MetaAgent
from evaluator_agent import EvaluatorAgent
from cleaning_agent import CleaningAgent
from base_agent import TaskAgentVariant
from config import get_config, validate_config


def create_sample_aircraft_states():
    """Create sample aircraft states for testing"""
    return {
        "all_aircraft": [
            {
                "aircraft_type": "A321",
                "lat": 40.649128,
                "lon": -73.815918,
                "speed": 15.2,
                "altitude": "ground",
                "heading": "090",
                "callsign": "JBU123",
                "aircraft_type_description": "Airbus A321",
                "flight_phase": "taxiing",
                "runway_proximity": "approaching 08L/26R",
                "airport_area": "north_field",
            },
            {
                "aircraft_type": "B737",
                "lat": 40.647984,
                "lon": -73.815887,
                "speed": 0.0,
                "altitude": "ground",
                "heading": "N/A",
                "callsign": "DAL456",
                "aircraft_type_description": "Boeing 737",
                "flight_phase": "parked/stationary",
                "runway_proximity": "distant from runways",
                "airport_area": "north_field",
            },
            {
                "aircraft_type": "A359",
                "lat": 40.646726,
                "lon": -73.809227,
                "speed": 8.5,
                "altitude": "ground",
                "heading": "270",
                "callsign": "AAL789",
                "aircraft_type_description": "Airbus A350-900",
                "flight_phase": "taxiing",
                "runway_proximity": "close to 08L/26R",
                "airport_area": "south_field",
            },
        ],
        "total_aircraft_count": 3,
        "timestamp": "2025-09-21T14:30:00Z",
    }


def create_sample_validation_record():
    """Create sample validation record for testing"""
    return {
        "record_id": "test_record_001",
        "timestamp_speech_start": "2025-09-21T14:30:00.000000",
        "timestamp_processing_complete": "2025-09-21T14:30:05.123456",
        "processing_lag_seconds": 5.123456,
        "aircraft_states": create_sample_aircraft_states(),
        "atc_command": {
            "command_type": "taxi_instruction",
            "target_aircraft": "JBU123",
            "details": "taxi via alpha to runway 08L",
            "reasoning": "Aircraft approaching runway needs taxi clearance",
            "confidence": 0.85,
        },
        "correlation_metadata": {
            "speech_segment_duration": 5.123456,
            "aircraft_state_interpolated": False,
            "timing_confidence": "high",
            "ground_aircraft_count": 3,
            "total_aircraft_count": 3,
        },
    }


async def test_task_agent():
    """Test Task Agent functionality"""
    print("\n🎯 Testing Task Agent...")

    config = get_config()

    # Create test variant
    test_variant = TaskAgentVariant(
        variant_id="test_variant_001",
        generation=0,
        parent_id=None,
        mutation_type="test",
        parameters={
            "model": "llama-3.3-70b",
            "temperature": 0.3,
            "max_tokens": 500,
            "gnn_hidden_dim": 64,
        },
    )

    # Initialize task agent
    task_agent = TaskAgent(config["cerebras_api_key"], test_variant)

    # Test with sample data
    aircraft_states = create_sample_aircraft_states()

    try:
        result = await task_agent.process({"aircraft_states": aircraft_states})

        print("✅ Task Agent Response:")
        print(
            f"   Action Type: {result.get('predicted_action', {}).get('command_type', 'Unknown')}"
        )
        print(
            f"   Target Aircraft: {result.get('predicted_action', {}).get('target_aircraft', 'None')}"
        )
        print(
            f"   Details: {result.get('predicted_action', {}).get('details', 'None')}"
        )
        print(f"   Confidence: {result.get('confidence', 0.0):.3f}")
        print(f"   Reasoning: {result.get('reasoning', 'No reasoning')[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Task Agent Error: {e}")
        return False


async def test_evaluator_agent():
    """Test Evaluator Agent functionality"""
    print("\n⚖️ Testing Evaluator Agent...")

    config = get_config()
    evaluator = EvaluatorAgent(config["cerebras_api_key"])

    # Sample predicted and expected actions
    predicted_action = {
        "command_type": "taxi_instruction",
        "target_aircraft": "JBU123",
        "details": "taxi via bravo to runway 08L",
    }

    expected_action = {
        "command_type": "taxi_instruction",
        "target_aircraft": "JBU123",
        "details": "taxi via alpha to runway 08L",
    }

    try:
        result = await evaluator.process(
            {
                "predicted_action": predicted_action,
                "expected_action": expected_action,
                "aircraft_states": create_sample_aircraft_states(),
            }
        )

        print("✅ Evaluator Agent Response:")
        print(f"   Score: {result.get('score', 0.0):.3f}")
        print(f"   Safety: {result.get('safety_assessment', 'Unknown')}")
        print(f"   Impact: {result.get('operational_impact', 'Unknown')}")
        print(f"   Reasoning: {result.get('reasoning', 'No reasoning')[:100]}...")

        return True

    except Exception as e:
        print(f"❌ Evaluator Agent Error: {e}")
        return False


async def test_cleaning_agent():
    """Test Cleaning Agent functionality"""
    print("\n🧹 Testing Cleaning Agent...")

    config = get_config()
    cleaner = CleaningAgent(config["cerebras_api_key"])

    # Test with sample validation record
    validation_record = create_sample_validation_record()

    try:
        result = await cleaner.process({"validation_record": validation_record})

        print("✅ Cleaning Agent Response:")
        print(
            f"   Decision: {'KEEP' if result.get('should_keep', False) else 'REJECT'}"
        )
        print(f"   Confidence: {result.get('confidence_score', 0.0):.3f}")
        print(f"   Reasoning: {result.get('reasoning', 'No reasoning')}")

        issues = result.get("quality_issues", [])
        if issues:
            print(f"   Quality Issues: {', '.join(issues)}")

        return True

    except Exception as e:
        print(f"❌ Cleaning Agent Error: {e}")
        return False


async def test_meta_agent():
    """Test Meta Agent functionality"""
    print("\n🧠 Testing Meta Agent...")

    config = get_config()
    meta_agent = MetaAgent(config["cerebras_api_key"])

    # Create sample current variants and performance data
    current_variants = [
        TaskAgentVariant(
            variant_id="test_variant_001",
            generation=1,
            parent_id=None,
            mutation_type="baseline",
            parameters={"model": "llama-3.3-70b", "temperature": 0.3},
            performance_score=0.75,
            evaluation_count=10,
        ),
        TaskAgentVariant(
            variant_id="test_variant_002",
            generation=1,
            parent_id=None,
            mutation_type="parameter_tuning",
            parameters={"model": "llama-3.3-70b", "temperature": 0.5},
            performance_score=0.68,
            evaluation_count=10,
        ),
    ]

    try:
        result = await meta_agent.process(
            {
                "current_variants": current_variants,
                "performance_data": [],  # Empty for this test
            }
        )

        new_variants = result.get("variants", [])

        print("✅ Meta Agent Response:")
        print(f"   Generated Variants: {len(new_variants)}")
        print(f"   Generation: {result.get('generation', 0)}")

        for i, variant in enumerate(new_variants[:3]):  # Show first 3
            print(f"   Variant {i+1}: {variant.variant_id} ({variant.mutation_type})")

        return True

    except Exception as e:
        print(f"❌ Meta Agent Error: {e}")
        return False


async def test_full_system():
    """Test minimal full system integration"""
    print("\n🚀 Testing Full ADAS System...")

    config = get_config()

    try:
        # Initialize ADAS system
        adas = ADASystem(
            cerebras_api_key=config["cerebras_api_key"],
            validation_dataset_path=config["paths"]["validation_dataset"],
        )

        print("✅ ADAS System initialized successfully")
        print(f"   Initial variants: {len(adas.current_variants)}")

        # Test loading validation data (even if empty)
        validation_records = adas._load_validation_data()
        print(f"   Validation records found: {len(validation_records)}")

        return True

    except Exception as e:
        print(f"❌ Full System Error: {e}")
        return False


async def main():
    """Run all tests"""
    print("🧪 ADAS SYSTEM BASIC TESTS")
    print("=" * 50)

    # Validate configuration first
    if not validate_config():
        print("❌ Configuration validation failed - cannot run tests")
        return

    print("✅ Configuration validated")

    # Run individual component tests
    tests = [
        ("Task Agent", test_task_agent),
        ("Evaluator Agent", test_evaluator_agent),
        ("Cleaning Agent", test_cleaning_agent),
        ("Meta Agent", test_meta_agent),
        ("Full System", test_full_system),
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = await test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False

    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)

    passed = sum(1 for result in results.values() if result)
    total = len(results)

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s} {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! ADAS system is ready.")
    else:
        print("⚠️  Some tests failed. Check configuration and API connectivity.")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
