# Informed Fast ATC Transcriber (IFA) - Implementation Vision

## 🎯 Core Mission

Create validation dataset for AI ATC agent by correlating live ATC transcriptions with real-time aircraft states at JFK airport.

## 🏗️ Architecture Philosophy

### Timing Correlation Strategy

- **Simplified Approach**: Fetch aircraft data when speech segment processing completes
- **Rationale**: API data reflects current state closely aligned with radio chatter timing
- **No Lag Compensation**: API calls synchronous with transcription completion
- **Key Insight**: Real ATC commands refer to current aircraft positions, not future states

### Data Collection Scope

- **Primary Focus**: All aircraft on ground at JFK (category: "ground")
- **Expansion Ready**: Comments indicate how to include approach/departure aircraft
- **Airport Specificity**: JFK-centric with runway/taxiway/gate knowledge
- **Validation Purpose**: State S1 → Action Y mapping for AI agent training

### Dataset Structure

- **Format**: JSON files (t1-<timestamp>.json, t2-<timestamp>.json, ...)
- **Batch Size**: 100 validation records per file
- **Record Structure**: {timestamp, aircraft_states, atc_command, command_type, affected_aircraft}
- **Metadata**: Processing confidence, correlation quality, JFK-specific context

## 🚀 Implementation Principles

### 1. Transcription Enhancement

- **Base System**: Leverage fastatc_transcriber.py proven architecture
- **JFK Context**: Enhanced system prompts with JFK runway/taxiway/gate information
- **Command Parsing**: Structured extraction of callsigns, actions, locations
- **Quality Control**: Confidence scoring and unclear transcription handling

### 2. Aircraft State Integration

- **Data Source**: client.py PlaneMonitor system (ADS-B via adsb.lol)
- **Update Frequency**: Synchronized with transcription completion events
- **State Snapshot**: Complete JFK ground aircraft positions at command time
- **Future Expansion**: Comments for including airborne aircraft filters

### 3. Command Categorization

- **Types**: taxi, takeoff, landing, frequency_change, hold, runway_crossing, altitude, speed, acknowledgment
- **Parsing**: Extract callsigns, frequencies, runways, taxiways from transcriptions
- **Filtering**: Log ignored communications (pure acknowledgments, weather, coordination)
- **JFK Specificity**: Enhanced with JFK runway/taxiway/gate knowledge

### 4. JFK Enhancement Features

- **Runway Occupancy**: Track which aircraft occupy which runways
- **Gate Assignment**: Monitor aircraft at gates vs taxiing
- **Hot Spot Monitoring**: Complex intersection awareness
- **Departure/Arrival Queues**: Sequential aircraft flow modeling
- **Weather Integration**: Future capability for weather impact correlation

## 🔧 Technical Implementation

### Core Components

1. **InformedATCTranscriber**: Main orchestrator class
2. **JFKContextualTranscriber**: Enhanced FastATCTranscriber with JFK knowledge
3. **AircraftStateManager**: Synchronized aircraft data collection
4. **ValidationDatasetBuilder**: Structured record creation and file management
5. **ATCCommandParser**: Command categorization and extraction

### Data Flow

1. Audio capture → Speech transcription (enhanced with JFK context)
2. Transcription completion → Aircraft state API call
3. Command parsing → Aircraft state correlation
4. Validation record creation → JSON dataset file
5. Batch management → File rotation every 100 records

### Expansion Points

- Aircraft filter expansion (ground → approach → departure)
- Multi-airport support (JFK → LGA, EWR, etc.)
- Real-time AI agent testing integration
- Advanced weather/traffic correlation

## 🎯 Success Metrics

- **Dataset Quality**: Clear command-state correlations for AI training
- **JFK Specificity**: Accurate runway/taxiway/gate context in all records
- **Processing Reliability**: Minimal data loss, high transcription confidence
- **AI Agent Readiness**: Structured format compatible with LLM training

## 🚨 Critical Notes for Future Development

- **Maintain JFK Focus**: All enhancements should prioritize JFK operational accuracy
- **Preserve Timing Logic**: API calls synchronized with transcription completion
- **Dataset Integrity**: Each record must contain complete state snapshot
- **Expansion Readiness**: Code structure supports multi-airport scaling
- **AI Agent Compatibility**: Dataset format must remain LLM-friendly

## 🔄 Development Iterations

1. **Phase 1**: Basic ground aircraft + transcription correlation
2. **Phase 2**: Enhanced JFK context and command parsing
3. **Phase 3**: Approach/departure aircraft integration
4. **Phase 4**: Multi-airport and weather correlation
5. **Phase 5**: Real-time AI agent validation testing

---

_This document serves as the north star for IFA development. All code changes should align with these principles and enhance the core mission of creating high-quality validation data for AI ATC agent training._
