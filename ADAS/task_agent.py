"""
TaskAgent for ADAS System

The TaskAgent responds to bottleneck situations with appropriate ATC actions.
It analyzes aircraft states and provides ATC commands to resolve bottlenecks.
"""

from typing import Dict, Any
from base_agent import BaseAgent, TaskAgentVariant


class TaskAgent(BaseAgent):
    """
    Task Agent: Responds to bottleneck situations with appropriate ATC actions
    This agent reads aircraft states and proposes ATC actions
    """

    def __init__(self, cerebras_api_key: str, variant: TaskAgentVariant):
        super().__init__(cerebras_api_key, agent_name="TaskAgent")
        self.variant = variant
        # ADAS uses pure LLM reasoning - no PyTorch models needed

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process aircraft states and predict ATC actions

        Args:
            input_data: Contains aircraft_states and context

        Returns:
            Dict with predicted ATC action, reasoning, and confidence
        """
        aircraft_states = input_data.get("aircraft_states", {})

        # Analyze aircraft states using specialized prompt
        analysis_prompt = self._build_analysis_prompt(aircraft_states)

        if not self.cerebras_client:
            return {
                "predicted_action": {
                    "command_type": "no_action",
                    "details": "Cerebras client unavailable",
                },
                "reasoning": "No AI client available",
                "confidence": 0.0,
            }

        try:
            # Global rate limiting
            await self.rate_limiter.wait_if_needed(self.agent_name)

            # Get TaskAgent specific model from config
            from config import get_config

            config = get_config()
            default_model = config["models"].get(
                "task_agent_model", "qwen-3-235b-a22b-thinking-2507"
            )

            response = self.cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": analysis_prompt}],
                model=self.variant.parameters.get("model", default_model),
                max_tokens=self.variant.parameters.get("max_tokens", 1000),
                temperature=self.variant.parameters.get("temperature", 0.3),
            )

            # Record successful API call
            self.rate_limiter.record_success()

            result_text = response.choices[0].message.content
            return self._parse_atc_response(result_text)

        except Exception as e:
            return {
                "predicted_action": {"command_type": "error", "details": str(e)},
                "reasoning": f"Error in AI processing: {e}",
                "confidence": 0.0,
            }

    def _build_analysis_prompt(self, aircraft_states: Dict[str, Any]) -> str:
        """Build analysis prompt for the task agent"""

        # Extract key information from aircraft states
        all_aircraft = aircraft_states.get("all_aircraft", [])
        total_aircraft = len(all_aircraft)

        # Analyze aircraft positions and movements
        ground_aircraft = [
            a for a in all_aircraft if a.get("flight_phase") == "parked/stationary"
        ]
        moving_aircraft = [a for a in all_aircraft if a.get("speed", 0) > 5]
        runway_proximity = [
            a for a in all_aircraft if "runway" in a.get("runway_proximity", "").lower()
        ]

        prompt = f"""You are an expert Air Traffic Controller AI analyzing airport operations at JFK.

CURRENT AIRPORT STATE:
- Total aircraft: {total_aircraft}
- Ground/stationary aircraft: {len(ground_aircraft)}
- Moving aircraft (>5 knots): {len(moving_aircraft)}
- Aircraft near runways: {len(runway_proximity)}

AIRCRAFT DETAILS:
"""

        # Add detailed aircraft information
        for i, aircraft in enumerate(
            all_aircraft[:20]
        ):  # Limit to first 20 for prompt length
            prompt += f"""
Aircraft {i+1}: {aircraft.get('callsign', 'Unknown').strip()}
- Type: {aircraft.get('aircraft_type', 'N/A')}
- Position: {aircraft.get('lat', 0):.6f}, {aircraft.get('lon', 0):.6f}
- Speed: {aircraft.get('speed', 0)} knots
- Area: {aircraft.get('airport_area', 'Unknown')}
- Runway proximity: {aircraft.get('runway_proximity', 'Unknown')}"""

        if total_aircraft > 20:
            prompt += f"\n... and {total_aircraft - 20} more aircraft"

        prompt += f"""

TASK: Analyze this airport state for potential bottlenecks and recommend ATC actions.

Consider:
1. Aircraft clustering that might cause delays
2. Runway approach queues forming
3. Taxiway congestion points
4. Gate area conflicts
5. Ground traffic flow issues

RESPONSE FORMAT:
Action Type: [taxi_instruction|runway_clearance|frequency_change|hold_instruction|no_action]
Target Aircraft: [callsign if specific aircraft targeted]
Command Details: [specific instruction details]
Reasoning: [why this action is needed]
Bottleneck Risk: [high|medium|low|none]
Confidence: [0.0-1.0]

Focus on proactive bottleneck prevention based on current aircraft positions and movements.
"""

        return prompt

    def _parse_atc_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI response into structured format"""
        lines = response_text.strip().split("\n")

        result = {
            "predicted_action": {
                "command_type": "no_action",
                "target_aircraft": None,
                "details": "",
            },
            "reasoning": "",
            "confidence": 0.5,
            "bottleneck_risk": "none",
        }

        for line in lines:
            line = line.strip()
            if line.startswith("Action Type:"):
                result["predicted_action"]["command_type"] = line.split(":", 1)[
                    1
                ].strip()
            elif line.startswith("Target Aircraft:"):
                result["predicted_action"]["target_aircraft"] = line.split(":", 1)[
                    1
                ].strip()
            elif line.startswith("Command Details:"):
                result["predicted_action"]["details"] = line.split(":", 1)[1].strip()
            elif line.startswith("Reasoning:"):
                result["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("Confidence:"):
                try:
                    result["confidence"] = float(line.split(":", 1)[1].strip())
                except:
                    result["confidence"] = 0.5
            elif line.startswith("Bottleneck Risk:"):
                result["bottleneck_risk"] = line.split(":", 1)[1].strip()

        return result
