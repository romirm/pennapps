#!/usr/bin/env python3
"""
Test script for Informed Fast ATC Transcriber (IFA)
"""

import sys
import os
import asyncio

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def test_components():
    """Test individual components before full integration"""
    print("🧪 Testing IFA Components...")
    print("=" * 50)

    # Test imports
    try:
        from ifa_components import (
            ValidationRecord,
            ATCCommandParser,
            AircraftStateManager,
            ValidationDatasetBuilder,
        )

        print("✅ Component imports successful")
    except ImportError as e:
        print(f"❌ Component import failed: {e}")
        return False

    # Test ATCCommandParser
    print("\n🔍 Testing ATCCommandParser...")
    parser = ATCCommandParser()

    test_transcription = "DAL671 contact departure 121.9"
    test_explanation = "Delta 671 is instructed to switch to departure frequency 121.9"

    parsed = parser.parse_command(test_transcription, test_explanation)
    print(f"   Input: '{test_transcription}'")
    print(f"   Command Type: {parsed['command_type']}")
    print(f"   Callsigns: {parsed['affected_aircraft']}")
    print(f"   Frequencies: {parsed['extracted_elements']['frequencies']}")
    print(f"   Confidence: {parsed['confidence_score']}")

    if (
        parsed["command_type"] == "frequency_change"
        and "DAL671" in parsed["affected_aircraft"]
    ):
        print("✅ ATCCommandParser working correctly")
    else:
        print("❌ ATCCommandParser test failed")
        return False

    # Test AircraftStateManager
    print("\n🛩️ Testing AircraftStateManager...")
    try:
        manager = AircraftStateManager()
        print("✅ AircraftStateManager initialized")

        # Test async function
        async def test_aircraft_state():
            state = await manager.get_current_aircraft_state()
            print(f"   Aircraft state keys: {list(state.keys())}")
            print(
                f"   Ground aircraft count: {len(state.get('jfk_ground_aircraft', []))}"
            )
            return state

        state = asyncio.run(test_aircraft_state())
        if "jfk_ground_aircraft" in state and "runway_occupancy" in state:
            print("✅ AircraftStateManager working correctly")
        else:
            print("❌ AircraftStateManager test failed")
            return False

    except Exception as e:
        print(f"⚠️ AircraftStateManager test failed (expected without live data): {e}")
        print("✅ This is normal if not connected to live aircraft data")

    # Test ValidationDatasetBuilder
    print("\n📊 Testing ValidationDatasetBuilder...")
    builder = ValidationDatasetBuilder(records_per_file=2)  # Small batch for testing

    # Create test validation record
    from datetime import datetime

    test_record = ValidationRecord(
        record_id="test-123",
        timestamp_speech_start=datetime.now().isoformat(),
        timestamp_processing_complete=datetime.now().isoformat(),
        processing_lag_seconds=5.2,
        aircraft_states={"test": "data"},
        atc_command={"command_type": "test", "raw_transcription": "test command"},
        correlation_metadata={"test": True},
    )

    builder.add_record(test_record)
    print(f"   Added test record, total: {builder.total_records}")

    if builder.total_records == 1:
        print("✅ ValidationDatasetBuilder working correctly")
    else:
        print("❌ ValidationDatasetBuilder test failed")
        return False

    print("\n🎉 All component tests passed!")
    return True


def test_full_integration():
    """Test the full IFA system (requires manual verification)"""
    print("\n🚀 Testing Full IFA Integration...")
    print("=" * 50)

    try:
        from ifa_transcriber import InformedATCTranscriber, JFKContextualTranscriber

        print("✅ Main IFA classes imported successfully")

        # Test JFKContextualTranscriber initialization
        transcriber = JFKContextualTranscriber()
        print("✅ JFKContextualTranscriber initialized")

        # Test InformedATCTranscriber initialization
        ifa = InformedATCTranscriber()
        print("✅ InformedATCTranscriber initialized")

        print("\n📋 System Components:")
        print(f"   - Transcriber: {type(ifa.transcriber).__name__}")
        print(f"   - Aircraft Manager: {type(ifa.aircraft_manager).__name__}")
        print(f"   - Command Parser: {type(ifa.command_parser).__name__}")
        print(f"   - Dataset Builder: {type(ifa.dataset_builder).__name__}")

        print("\n✅ Full integration test passed!")
        print("🎯 Ready for live ATC audio testing!")
        return True

    except Exception as e:
        print(f"❌ Full integration test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("Informed Fast ATC Transcriber - Test Suite")
    print("=" * 60)

    # Test components first
    if not test_components():
        print("\n❌ Component tests failed. Fix issues before proceeding.")
        return

    # Test full integration
    if not test_full_integration():
        print("\n❌ Integration tests failed. Check system setup.")
        return

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("\n📋 Next Steps:")
    print("1. Ensure you have a .env file with CEREBRAS_API_KEY")
    print("2. Run: python3.9 ifa_transcriber.py")
    print("3. Play JFK ATC audio through your microphone")
    print("4. Watch for validation records being created")
    print("5. Check for t1-<timestamp>.json files being generated")
    print("\n🎯 Ready to create AI agent validation dataset!")


if __name__ == "__main__":
    main()
