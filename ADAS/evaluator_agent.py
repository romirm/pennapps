"""
EvaluatorAgent for ADAS System

The EvaluatorAgent acts as an LLM-as-a-judge for assessing task agent performance.
It compares predicted actions against expected actions from validation data.
"""

import json
from typing import Dict, Any

from base_agent import BaseAgent


class EvaluatorAgent(BaseAgent):
    """
    Evaluator Agent: LLM-as-a-judge for assessing task agent performance
    Compares predicted actions against expected actions from validation data
    """

    def __init__(self, cerebras_api_key: str):
        super().__init__(cerebras_api_key, agent_name="EvaluatorAgent")

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task agent performance on validation data"""

        predicted_action = input_data.get("predicted_action", {})
        expected_action = input_data.get("expected_action", {})
        aircraft_states = input_data.get("aircraft_states", {})

        evaluation_prompt = self._build_evaluation_prompt(
            predicted_action, expected_action, aircraft_states
        )

        if not self.cerebras_client:
            return {"score": 0.5, "reasoning": "No Cerebras client available"}

        try:
            # Global rate limiting
            await self.rate_limiter.wait_if_needed(self.agent_name)

            # Import config to get EvaluatorAgent specific model
            from config import get_config

            config = get_config()
            evaluator_model = config["models"].get(
                "evaluator_agent_model", "llama-4-scout-17b-16e-instruct"
            )

            response = self.cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": evaluation_prompt}],
                model=evaluator_model,
                max_tokens=1500,
                temperature=0.2,
            )

            # Record successful API call
            self.rate_limiter.record_success()

            result_text = response.choices[0].message.content

            # Debug: Save raw EvaluatorAgent response
            import os
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
            os.makedirs(tmp_dir, exist_ok=True)

            debug_file = os.path.join(
                tmp_dir, f"evaluator_raw_response_{timestamp}.txt"
            )
            with open(debug_file, "w") as f:
                f.write("EvaluatorAgent Raw Response\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"PREDICTED: {predicted_action}\n\n")
                f.write(f"EXPECTED: {expected_action}\n\n")
                f.write("RAW RESPONSE:\n")
                f.write(result_text)

            evaluation_result = self._parse_evaluation_response(result_text)

            # Print detailed reasoning for transparency
            self._print_evaluation_reasoning(
                evaluation_result, predicted_action, expected_action
            )

            print(f"🔍 Debug: Raw response saved to {os.path.basename(debug_file)}")

            return evaluation_result

        except Exception as e:
            # Record failed API call
            self.rate_limiter.record_error()
            return {"score": 0.0, "reasoning": f"Error: {e}"}

    def _build_evaluation_prompt(
        self,
        predicted_action: Dict[str, Any],
        expected_action: Dict[str, Any],
        aircraft_states: Dict[str, Any],
    ) -> str:
        """Build evaluation prompt comparing actions"""

        prompt = f"""You are an expert ATC evaluation judge assessing AI agent performance.

AIRCRAFT STATE CONTEXT:
{json.dumps(aircraft_states, indent=2)}

PREDICTED ACTION (by AI agent):
{json.dumps(predicted_action, indent=2)}

EXPECTED ACTION (from real ATC transcript):
{json.dumps(expected_action, indent=2)}

EVALUATION TASK:
Compare the AI agent's predicted action against the real ATC action. Provide detailed analysis covering:

EVALUATION CRITERIA:
1. CORRECTNESS: Does the predicted action address the same operational need as the expected action?
2. SAFETY: Is the predicted action safe and appropriate for the traffic situation?
3. EFFICIENCY: Would this action help or hinder traffic flow compared to the expected action?
4. SPECIFICITY: Are aircraft callsigns, altitudes, headings, and instructions specific enough?
5. TIMING: Is this the right time for this type of action given the aircraft states?
6. ALTERNATIVES: Could the predicted action be a valid alternative to the expected action?

SCORING SCALE:
- 1.0: Perfect match or equally valid alternative action that achieves the same goal
- 0.8-0.9: Very good, minor differences in approach but same operational intent
- 0.6-0.7: Good, addresses same issue but with different method or timing
- 0.4-0.5: Partially correct, misses some key aspects but shows understanding
- 0.2-0.3: Poor, incorrect approach, timing, or misunderstands the situation
- 0.0-0.1: Completely wrong, unsafe, or shows no understanding

DETAILED ANALYSIS REQUIRED:
- Compare the specific commands (altitude changes, heading changes, speed, etc.)
- Analyze if both actions serve the same traffic management purpose
- Consider if the predicted action would create safety issues
- Evaluate whether the predicted action shows understanding of ATC procedures
- Assess the appropriateness of the action timing

RESPONSE FORMAT:
Score: [0.0-1.0]
Reasoning: [Provide detailed step-by-step comparison explaining: 1) What the expected action was trying to achieve, 2) What the predicted action would achieve, 3) How well they align, 4) Any safety or efficiency concerns, 5) Whether the predicted action demonstrates good ATC judgment]
Safety Assessment: [safe|questionable|unsafe]
Operational Impact: [positive|neutral|negative]
"""

        return prompt

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse evaluation response"""

        result = {
            "score": 0.5,
            "reasoning": "",
            "safety_assessment": "questionable",
            "operational_impact": "neutral",
        }

        lines = response_text.strip().split("\n")

        # Extract reasoning - everything between "Reasoning:" and safety assessment
        reasoning_lines = []
        in_reasoning = False

        for line in lines:
            line = line.strip()

            if line.startswith("Score:"):
                try:
                    score_text = line.split(":", 1)[1].strip()
                    result["score"] = float(score_text)
                except:
                    result["score"] = 0.5
            elif line.startswith("Reasoning:"):
                in_reasoning = True
                reasoning_text = line.split(":", 1)[1].strip()
                if reasoning_text:  # If there's text after the colon
                    reasoning_lines.append(reasoning_text)
            elif line.startswith("**Safety Assessment**:") or line.startswith(
                "Safety Assessment:"
            ):
                in_reasoning = False
                result["safety_assessment"] = line.split(":", 1)[1].strip()
            elif line.startswith("**Operational Impact**:") or line.startswith(
                "Operational Impact:"
            ):
                result["operational_impact"] = line.split(":", 1)[1].strip()
            elif in_reasoning and line:
                reasoning_lines.append(line)

        # Join all reasoning lines
        if reasoning_lines:
            result["reasoning"] = " ".join(reasoning_lines)

        return result

    def _print_evaluation_reasoning(
        self,
        evaluation_result: Dict[str, Any],
        predicted_action: Dict[str, Any],
        expected_action: Dict[str, Any],
    ):
        """Print detailed evaluation reasoning for transparency"""

        score = evaluation_result.get("score", 0.0)
        reasoning = evaluation_result.get("reasoning", "No reasoning provided")
        safety = evaluation_result.get("safety_assessment", "unknown")
        impact = evaluation_result.get("operational_impact", "unknown")

        print(f"\n📊 EVALUATOR REASONING:")
        print(f"   Score: {score:.3f}/1.0")
        print(f"   Safety: {safety}")
        print(f"   Impact: {impact}")

        print(
            f"\n🤖 PREDICTED: {predicted_action.get('command_type', 'unknown')} - {predicted_action.get('details', 'N/A')}"
        )
        print(
            f"✅ EXPECTED:  {expected_action.get('command_type', 'unknown')} - {expected_action.get('details', 'N/A')}"
        )

        print(f"\n💭 REASONING: {reasoning}")

        # Add score interpretation
        if score >= 0.8:
            print("🟢 Excellent performance - Very close to expected action")
        elif score >= 0.6:
            print("🟡 Good performance - Addresses the same operational need")
        elif score >= 0.4:
            print("🟠 Fair performance - Partially correct approach")
        elif score >= 0.2:
            print("🔴 Poor performance - Incorrect approach")
        else:
            print("❌ Very poor performance - Completely wrong or unsafe")

        print("-" * 60)
