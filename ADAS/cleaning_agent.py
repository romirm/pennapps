"""
CleaningAgent for ADAS System

The CleaningAgent filters and cleans validation datasets, removing low-confidence,
vague, or invalid records to improve training data quality.
"""

import json
from typing import Dict, Any

from base_agent import BaseAgent


class CleaningAgent(BaseAgent):
    """
    Cleaning Agent: Filters and cleans validation dataset
    Removes low-confidence, vague, or invalid records
    """

    def __init__(self, cerebras_api_key: str):
        super().__init__(cerebras_api_key, agent_name="CleaningAgent")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and filter validation records"""

        raw_record = input_data.get("validation_record", {})

        cleaning_prompt = self._build_cleaning_prompt(raw_record)

        if not self.cerebras_client:
            return {"should_keep": True, "reasoning": "No Cerebras client available"}

        try:
            # Global rate limiting
            await self.rate_limiter.wait_if_needed(self.agent_name)

            # Import config to get default model (CleaningAgent uses default)
            from config import get_config

            config = get_config()
            default_model = config["models"].get("default_model", "llama3.1-8b")

            response = self.cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": cleaning_prompt}],
                model=default_model,
                max_tokens=800,
                temperature=0.2,
            )

            # Record successful API call
            self.rate_limiter.record_success()

            result_text = response.choices[0].message.content
            return self._parse_cleaning_response(result_text)

        except Exception as e:
            # Record failed API call
            self.rate_limiter.record_error()
            return {"should_keep": False, "reasoning": f"Error: {e}"}

    def _build_cleaning_prompt(self, record: Dict[str, Any]) -> str:
        """Build cleaning evaluation prompt for selecting high-quality training examples"""

        # Extract key information for analysis
        atc_command = record.get("atc_command", {})
        raw_transcription = atc_command.get("raw_transcription", "")
        processed_explanation = atc_command.get("processed_explanation", "")
        command_type = atc_command.get("command_type", "unknown")
        confidence_score = atc_command.get("confidence_score", 0.0)
        affected_aircraft = atc_command.get("affected_aircraft", [])
        extracted_elements = atc_command.get("extracted_elements", {})

        aircraft_states = record.get("aircraft_states", {})
        total_aircraft = aircraft_states.get("total_aircraft_count", 0)

        prompt = f"""You are an expert ATC data curator selecting the BEST training examples for an AI system.

VALIDATION RECORD ANALYSIS:
Raw Transcription: "{raw_transcription}"
Command Type: {command_type}
Confidence Score: {confidence_score}
Total Aircraft in Scene: {total_aircraft}
Affected Aircraft: {affected_aircraft}
Extracted Elements: {json.dumps(extracted_elements, indent=2)}

PROCESSED EXPLANATION QUALITY:
{processed_explanation[:500]}{"..." if len(processed_explanation) > 500 else ""}

SELECTION CRITERIA (prioritize HIGH-QUALITY examples):

🏆 EXCELLENT CANDIDATES (KEEP):
- Clear, specific ATC commands (taxi, takeoff, landing clearances)
- High confidence scores (>0.7)
- Identifiable aircraft callsigns
- Operational relevance (runway assignments, frequency changes)
- Rich context with multiple aircraft
- Complete extracted elements (runways, taxiways, frequencies)

❌ POOR CANDIDATES (REJECT):
- Fragments or incomplete transmissions
- Low confidence scores (<0.5)
- Unknown command types with no clear action
- Missing callsigns or aircraft identification
- Administrative chatter or non-operational talk
- Garbled or unclear transcriptions
- Empty or minimal extracted elements

🎯 TRAINING VALUE ASSESSMENT:
- Would this example teach an AI agent about real ATC operations?
- Does it contain actionable information for bottleneck resolution?
- Is the command clear enough for performance evaluation?
- Does it represent typical ATC communication patterns?

RESPONSE FORMAT:
Decision: [KEEP|REJECT]
Quality Score: [0.0-1.0] (training value)
Training Value: [excellent|good|fair|poor]
Primary Reason: [why this example is valuable or should be rejected]
Key Strengths: [what makes this a good training example]
Issues Found: [any problems that reduce training value]
"""

        return prompt

    def _parse_cleaning_response(self, response_text: str) -> Dict[str, Any]:
        """Parse cleaning decision response with enhanced training value assessment"""

        result = {
            "should_keep": True,
            "quality_score": 0.5,
            "training_value": "fair",
            "reasoning": "",
            "key_strengths": [],
            "issues_found": [],
        }

        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("Decision:"):
                decision = line.split(":", 1)[1].strip()
                result["should_keep"] = decision.upper() == "KEEP"
            elif line.startswith("Quality Score:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    result["quality_score"] = float(score_text)
                except:
                    result["quality_score"] = 0.5
            elif line.startswith("Training Value:"):
                result["training_value"] = line.split(":", 1)[1].strip()
            elif line.startswith("Primary Reason:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("Key Strengths:"):
                strengths_text = line.split(":", 1)[1].strip()
                result["key_strengths"] = [
                    s.strip() for s in strengths_text.split(",") if s.strip()
                ]
            elif line.startswith("Issues Found:"):
                issues_text = line.split(":", 1)[1].strip()
                result["issues_found"] = [
                    i.strip() for i in issues_text.split(",") if i.strip()
                ]

        return result
