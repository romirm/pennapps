"""
MetaAgent for ADAS System

The MetaAgent proposes mutations and variants of task agents, driving the evolutionary
process of improving task agents through strategic mutations.
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import asdict

from base_agent import BaseAgent, TaskAgentVariant, EvaluationResult


class MetaAgent(BaseAgent):
    """
    ADAS Meta Agent: Designs and implements new task agents as code

    Following the ADAS research paper methodology:
    1. Design + Implement: Drafts high-level ideas and implements them as code
    2. Self-reflection: Ensures novelty and debugs code issues
    3. Evaluate → Archive → Iterate: Conditions future proposals on archive

    The meta agent writes concrete code artifacts (forward() functions and supporting logic),
    not just parameter tweaks. It's a meta-improving-target architecture where the meta
    agent improves separate task agents, not itself.
    """

    def __init__(self, cerebras_api_key: str):
        # MetaAgent uses qwen-3-coder-480b which has 10 requests/minute limit
        super().__init__(cerebras_api_key, agent_name="MetaAgent")
        self.generation_count = 0
        self.design_archive = []  # Archive of designs and their performance
        self.reflection_max_attempts = 3  # Max reflection attempts for debugging

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        ADAS Process: Design and implement new task agents as code artifacts

        Following the ADAS methodology:
        1. Design + Implement: Create concrete code proposals
        2. Self-reflection: Ensure novelty and debug issues
        3. Archive: Store designs and performance for future conditioning
        """

        current_variants = input_data.get("current_variants", [])
        performance_data = input_data.get("performance_data", [])

        # Update archive with performance data
        self._update_design_archive(current_variants, performance_data)

        if not self.cerebras_client:
            return {"variants": [], "reasoning": "No Cerebras client available"}

        try:
            self.generation_count += 1

            # Step 1: Design + Implement - Generate code proposals
            design_proposals = await self._generate_design_proposals()

            # Step 2: Self-reflection - Ensure novelty and debug
            refined_proposals = await self._self_reflect_on_designs(design_proposals)

            # Step 3: Convert to TaskAgent variants
            new_variants = self._convert_to_variants(refined_proposals)

            # Save proposals to tmp folder for inspection
            self._save_design_proposals(refined_proposals)

            return {
                "variants": new_variants,
                "generation": self.generation_count,
                "reasoning": f"Generated {len(new_variants)} code-based agent designs using ADAS methodology",
                "design_proposals": len(design_proposals),
                "refined_proposals": len(refined_proposals),
            }

        except Exception as e:
            # Record failed API call
            self.rate_limiter.record_error()
            print(f"❌ {self.agent_name} API error: {e}")
            return {"variants": [], "reasoning": f"Error: {e}"}

    def _update_design_archive(
        self,
        current_variants: List[TaskAgentVariant],
        performance_data: List[EvaluationResult],
    ):
        """Update the design archive with performance data for future conditioning"""

        # Group performance data by variant
        variant_performance = {}
        for result in performance_data:
            if result.variant_id not in variant_performance:
                variant_performance[result.variant_id] = []
            variant_performance[result.variant_id].append(result.score)

        # Update archive with average performance
        for variant in current_variants:
            if variant.variant_id in variant_performance:
                scores = variant_performance[variant.variant_id]
                avg_score = sum(scores) / len(scores) if scores else 0.0

                # Store design in archive
                design_entry = {
                    "variant_id": variant.variant_id,
                    "generation": variant.generation,
                    "mutation_type": variant.mutation_type,
                    "parameters": variant.parameters.copy(),
                    "performance_score": avg_score,
                    "evaluation_count": len(scores),
                    "design_concept": self._extract_design_concept(variant),
                }

                # Update existing entry or add new one
                existing_idx = next(
                    (
                        i
                        for i, entry in enumerate(self.design_archive)
                        if entry["variant_id"] == variant.variant_id
                    ),
                    None,
                )

                if existing_idx is not None:
                    self.design_archive[existing_idx] = design_entry
                else:
                    self.design_archive.append(design_entry)

        # Keep archive manageable (top 20 designs)
        self.design_archive.sort(key=lambda x: x["performance_score"], reverse=True)
        self.design_archive = self.design_archive[:20]

    def _extract_design_concept(self, variant: TaskAgentVariant) -> str:
        """Extract the high-level design concept from a variant"""

        prompt_style = variant.parameters.get("prompt_style", "structured")
        analysis_approach = variant.parameters.get("analysis_approach", "comprehensive")
        # Get the configured task agent model as fallback
        from config import get_config

        config = get_config()
        default_task_model = config["models"].get(
            "task_agent_model", "qwen-3-235b-a22b-thinking-2507"
        )
        model = variant.parameters.get("model", default_task_model)

        return f"{analysis_approach}_{prompt_style}_approach_with_{model.replace('-', '_')}"

    async def _generate_design_proposals(self) -> List[Dict[str, Any]]:
        """Generate new design proposals as concrete code artifacts"""

        design_prompt = self._build_design_prompt()

        # Global rate limiting
        await self.rate_limiter.wait_if_needed(self.agent_name)

        # Import config to get meta agent model
        from config import get_config

        config = get_config()
        meta_model = config["models"].get("meta_agent_model", "qwen-3-coder-480b")

        response = self.cerebras_client.chat.completions.create(
            messages=[{"role": "user", "content": design_prompt}],
            model=meta_model,
            max_tokens=4000,  # Larger for code generation
            temperature=0.8,  # Higher for creativity
        )

        # Record successful API call
        self.rate_limiter.record_success()

        result_text = response.choices[0].message.content

        # Debug: Save raw MetaAgent response for inspection
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        debug_file = os.path.join(
            tmp_dir,
            f"meta_agent_raw_response_gen{self.generation_count}_{timestamp}.txt",
        )
        with open(debug_file, "w") as f:
            f.write(f"Generation {self.generation_count} MetaAgent Raw Response\n")
            f.write("=" * 60 + "\n\n")
            f.write(result_text)

        proposals = self._parse_design_proposals(result_text)
        print(f"🔍 MetaAgent Debug: Generated {len(proposals)} proposals from response")
        print(f"   Raw response saved to: {os.path.basename(debug_file)}")

        return proposals

    def _build_design_prompt(self) -> str:
        """Build ADAS-style design prompt for generating code-based agent proposals"""

        prompt = f"""You are an ADAS Meta-Agent that designs and implements new ATC AI agents as concrete code artifacts.

GENERATION: {self.generation_count}

DESIGN ARCHIVE (Previous Designs & Performance):"""

        # Include archive for conditioning future proposals
        if self.design_archive:
            for i, design in enumerate(self.design_archive[:10]):  # Top 10
                prompt += f"""
Design {i+1}: {design['design_concept']}
  - Performance: {design['performance_score']:.3f}
  - Approach: {design['parameters'].get('analysis_approach', 'N/A')}
  - Style: {design['parameters'].get('prompt_style', 'N/A')}
  - Model: {design['parameters'].get('model', 'N/A')}"""
        else:
            prompt += "\n(No previous designs in archive)"

        prompt += f"""

ADAS METHODOLOGY:
You must follow the ADAS research paper approach:

1. DESIGN + IMPLEMENT: Create concrete code proposals, not just parameter tweaks
2. FOCUS ON WORKFLOW: Design the agent's forward() function and reasoning process  
3. NOVEL APPROACHES: Ensure designs are meaningfully different from archive
4. CODE ARTIFACTS: Provide actual implementation details, not abstract ideas

TASK: Design 3-5 new ATC task agent workflows as code-based proposals.

Each proposal should define:
- High-level design concept and reasoning approach
- Specific workflow steps (planning, analysis, decision-making)
- Prompt engineering strategy and structure
- Tool usage and information processing
- Error handling and fallback mechanisms

RESPONSE FORMAT:
For each design proposal, provide:

PROPOSAL_N:
Design Concept: [Brief conceptual description]
Workflow Type: [multi_step_reasoning|reactive_analysis|hierarchical_planning|ensemble_approach|other]
Core Innovation: [What makes this design novel/different]
Forward Function Pseudocode:
```python
async def forward(self, aircraft_states, bottleneck_info):
    # Step 1: [Description]
    # ... detailed workflow steps
    # Step N: Return structured decision
    pass
```
Prompt Strategy: [How prompts are structured and used]
Model Requirements: [Specific model needs/preferences]
Expected Strengths: [What this design should excel at]
Potential Weaknesses: [Known limitations to address]

DESIGN PRINCIPLES:
- Create workflows that are fundamentally different from existing archive entries
- Focus on the PROCESS of decision-making, not just parameters
- Consider multi-step reasoning, tool integration, and error recovery
- Design for the specific domain: ATC bottleneck resolution
- Balance novelty with practical effectiveness

Generate designs that represent different philosophical approaches to ATC decision-making:
- Analytical vs. intuitive processing
- Sequential vs. parallel analysis  
- Conservative vs. aggressive optimization
- Local vs. global situation awareness
- Rule-based vs. learned pattern recognition
"""

        return prompt

    def _parse_design_proposals(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse design proposals from meta-agent response"""

        proposals = []
        current_proposal = None
        in_code_block = False
        code_lines = []

        lines = response_text.split("\n")

        for line in lines:
            line = line.strip()

            # Start of new proposal
            if line.startswith("PROPOSAL_"):
                if current_proposal:
                    proposals.append(current_proposal)

                current_proposal = {
                    "proposal_id": line.split(":")[0] if ":" in line else line,
                    "design_concept": "",
                    "workflow_type": "multi_step_reasoning",
                    "core_innovation": "",
                    "forward_function": "",
                    "prompt_strategy": "",
                    "model_requirements": "",
                    "expected_strengths": "",
                    "potential_weaknesses": "",
                }

            elif current_proposal:
                # Parse proposal fields
                if line.startswith("Design Concept:"):
                    current_proposal["design_concept"] = line.split(":", 1)[1].strip()
                elif line.startswith("Workflow Type:"):
                    current_proposal["workflow_type"] = line.split(":", 1)[1].strip()
                elif line.startswith("Core Innovation:"):
                    current_proposal["core_innovation"] = line.split(":", 1)[1].strip()
                elif line.startswith("Prompt Strategy:"):
                    current_proposal["prompt_strategy"] = line.split(":", 1)[1].strip()
                elif line.startswith("Model Requirements:"):
                    current_proposal["model_requirements"] = line.split(":", 1)[
                        1
                    ].strip()
                elif line.startswith("Expected Strengths:"):
                    current_proposal["expected_strengths"] = line.split(":", 1)[
                        1
                    ].strip()
                elif line.startswith("Potential Weaknesses:"):
                    current_proposal["potential_weaknesses"] = line.split(":", 1)[
                        1
                    ].strip()

                # Handle code blocks
                elif line.startswith("```python") or line.startswith("```"):
                    in_code_block = True
                    code_lines = []
                elif line.startswith("```") and in_code_block:
                    in_code_block = False
                    current_proposal["forward_function"] = "\n".join(code_lines)
                elif in_code_block:
                    code_lines.append(line)

        # Add last proposal
        if current_proposal:
            proposals.append(current_proposal)

        return proposals

    async def _self_reflect_on_designs(
        self, design_proposals: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Self-reflection to ensure novelty and debug code issues"""

        refined_proposals = []

        for proposal in design_proposals:
            # Reflection attempt counter
            attempts = 0
            current_proposal = proposal.copy()

            while attempts < self.reflection_max_attempts:
                # Check for novelty against archive
                is_novel = self._check_novelty(current_proposal)

                if is_novel:
                    # Add reflection metadata
                    current_proposal["reflection_attempts"] = attempts
                    current_proposal["novelty_verified"] = True
                    refined_proposals.append(current_proposal)
                    break
                else:
                    # Reflect and refine
                    current_proposal = await self._reflect_and_refine(
                        current_proposal, attempts
                    )
                    attempts += 1

            # If max attempts reached, still include but mark as potentially non-novel
            if attempts >= self.reflection_max_attempts:
                current_proposal["reflection_attempts"] = attempts
                current_proposal["novelty_verified"] = False
                refined_proposals.append(current_proposal)

        return refined_proposals

    def _check_novelty(self, proposal: Dict[str, Any]) -> bool:
        """Check if a design proposal is sufficiently novel compared to archive"""

        if not self.design_archive:
            return True  # First designs are always novel

        design_concept = proposal.get("design_concept", "").lower()
        workflow_type = proposal.get("workflow_type", "").lower()
        core_innovation = proposal.get("core_innovation", "").lower()

        # Check against existing designs
        for archived_design in self.design_archive:
            archived_concept = archived_design.get("design_concept", "").lower()

            # Simple similarity check (could be enhanced with embeddings)
            concept_words = set(design_concept.split())
            archived_words = set(archived_concept.split())

            # If >70% word overlap, consider not novel
            if concept_words and archived_words:
                overlap = len(concept_words.intersection(archived_words))
                similarity = overlap / len(concept_words.union(archived_words))

                if similarity > 0.7:
                    return False

        return True

    async def _reflect_and_refine(
        self, proposal: Dict[str, Any], attempt: int
    ) -> Dict[str, Any]:
        """Reflect on and refine a design proposal to increase novelty"""

        reflection_prompt = f"""You are reflecting on an ATC agent design proposal to ensure novelty and correctness.

ORIGINAL PROPOSAL:
Design Concept: {proposal.get('design_concept', '')}
Workflow Type: {proposal.get('workflow_type', '')}
Core Innovation: {proposal.get('core_innovation', '')}

EXISTING ARCHIVE CONCEPTS:
{[design['design_concept'] for design in self.design_archive[:5]]}

REFLECTION TASK (Attempt {attempt + 1}/{self.reflection_max_attempts}):
The original proposal may be too similar to existing designs. Please refine it to be more novel while maintaining effectiveness.

Focus on:
1. Making the core innovation more distinctive
2. Introducing unique workflow elements
3. Addressing different aspects of ATC decision-making
4. Ensuring the approach is genuinely different

Provide a refined version:

REFINED_PROPOSAL:
Design Concept: [More novel concept]
Workflow Type: [Potentially different workflow type]
Core Innovation: [More distinctive innovation]
Forward Function Pseudocode:
```python
# Refined implementation
```
Prompt Strategy: [Updated strategy]
Model Requirements: [Updated requirements]
Expected Strengths: [Updated strengths]
Potential Weaknesses: [Updated weaknesses]
"""

        try:
            # Global rate limiting
            await self.rate_limiter.wait_if_needed(self.agent_name)

            # Import config to get meta agent model
            from config import get_config

            config = get_config()
            meta_model = config["models"].get("meta_agent_model", "qwen-3-coder-480b")

            response = self.cerebras_client.chat.completions.create(
                messages=[{"role": "user", "content": reflection_prompt}],
                model=meta_model,
                max_tokens=3000,
                temperature=0.9,  # High creativity for refinement
            )

            # Record successful API call
            self.rate_limiter.record_success()

            result_text = response.choices[0].message.content

            # Parse refined proposal
            refined_proposals = self._parse_design_proposals(result_text)

            if refined_proposals:
                refined = refined_proposals[0]
                refined["original_proposal"] = proposal
                refined["refinement_attempt"] = attempt + 1
                return refined
            else:
                # If parsing fails, return original with attempt marker
                proposal["refinement_attempt"] = attempt + 1
                return proposal

        except Exception as e:
            print(f"⚠️ Reflection attempt {attempt + 1} failed: {e}")
            proposal["refinement_attempt"] = attempt + 1
            return proposal

    def _convert_to_variants(
        self, refined_proposals: List[Dict[str, Any]]
    ) -> List[TaskAgentVariant]:
        """Convert design proposals to TaskAgent variants"""

        variants = []

        for i, proposal in enumerate(refined_proposals):
            # Extract parameters from proposal
            workflow_type = proposal.get("workflow_type", "multi_step_reasoning")
            model_reqs = proposal.get("model_requirements", "").lower()

            # Map workflow type to implementation parameters
            parameters = self._map_workflow_to_parameters(workflow_type, model_reqs)

            # Add proposal-specific metadata
            parameters["design_concept"] = proposal.get("design_concept", "")
            parameters["core_innovation"] = proposal.get("core_innovation", "")
            parameters["workflow_type"] = workflow_type
            parameters["forward_function"] = proposal.get("forward_function", "")
            parameters["prompt_strategy"] = proposal.get("prompt_strategy", "")

            variant = TaskAgentVariant(
                variant_id=f"gen{self.generation_count}_{i+1}",
                generation=self.generation_count,
                parent_id=None,  # ADAS designs are novel, not mutations
                mutation_type="adas_design",  # Mark as ADAS-generated
                parameters=parameters,
                created_at=datetime.now().isoformat(),
            )

            variants.append(variant)

        # Generate and save TaskAgent Python files for each variant
        self._save_variants_as_python_files(variants)

        return variants

    def _save_variants_as_python_files(self, variants: List[TaskAgentVariant]):
        """Generate and save complete Python files for each TaskAgent variant"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        saved_files = []

        for variant in variants:
            # Generate and save the Python file
            self._save_task_agent_code(variant, tmp_dir, timestamp)
            saved_files.append(f"task_agent_{variant.variant_id}_{timestamp}.py")

        print(f"🐍 Generated {len(saved_files)} TaskAgent Python files:")
        for filename in saved_files:
            print(f"   {filename}")

        return saved_files

    def _map_workflow_to_parameters(
        self, workflow_type: str, model_reqs: str
    ) -> Dict[str, Any]:
        """Map workflow type to implementation parameters"""

        # Base parameters
        parameters = {
            "temperature": 0.3,
            "max_tokens": 1000,
            "analysis_approach": "comprehensive",
            "prompt_style": "structured",
        }

        # Workflow-specific mappings
        if workflow_type == "multi_step_reasoning":
            parameters.update(
                {
                    "analysis_approach": "comprehensive",
                    "prompt_style": "structured",
                    "temperature": 0.2,  # Lower for systematic reasoning
                    "max_tokens": 1500,
                }
            )
        elif workflow_type == "reactive_analysis":
            parameters.update(
                {
                    "analysis_approach": "rapid",
                    "prompt_style": "conversational",
                    "temperature": 0.4,  # Higher for reactive responses
                    "max_tokens": 800,
                }
            )
        elif workflow_type == "hierarchical_planning":
            parameters.update(
                {
                    "analysis_approach": "predictive",
                    "prompt_style": "technical",
                    "temperature": 0.3,
                    "max_tokens": 1200,
                }
            )
        elif workflow_type == "ensemble_approach":
            parameters.update(
                {
                    "analysis_approach": "risk_aware",
                    "prompt_style": "hybrid",
                    "temperature": 0.35,
                    "max_tokens": 1400,
                }
            )

        # Get configured models for proper selection
        from config import get_config

        config = get_config()
        default_task_model = config["models"].get(
            "task_agent_model", "qwen-3-235b-a22b-thinking-2507"
        )

        # Model selection based on requirements
        if "large" in model_reqs or "capable" in model_reqs:
            parameters["model"] = default_task_model  # Use configured task agent model
        elif "fast" in model_reqs or "efficient" in model_reqs:
            parameters["model"] = "llama3.1-8b"
        elif "balanced" in model_reqs:
            parameters["model"] = "llama-3.3-70b"
        else:
            parameters["model"] = default_task_model  # Use configured default

        return parameters

    def _save_design_proposals(self, refined_proposals: List[Dict[str, Any]]):
        """Save design proposals to tmp folder for inspection"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tmp_dir = os.path.join(os.path.dirname(__file__), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)

        # Save detailed proposals
        proposals_file = os.path.join(
            tmp_dir,
            f"adas_design_proposals_gen{self.generation_count}_{timestamp}.json",
        )

        proposals_data = {
            "generation": self.generation_count,
            "timestamp": timestamp,
            "methodology": "ADAS - Automated Design of Agentic Systems",
            "archive_size": len(self.design_archive),
            "proposals": refined_proposals,
        }

        with open(proposals_file, "w") as f:
            json.dump(proposals_data, f, indent=2)

        # Save archive state
        archive_file = os.path.join(
            tmp_dir, f"adas_design_archive_gen{self.generation_count}_{timestamp}.json"
        )

        with open(archive_file, "w") as f:
            json.dump(self.design_archive, f, indent=2)

        print(f"💾 ADAS Design Proposals saved to tmp/:")
        print(f"   Proposals: {os.path.basename(proposals_file)}")
        print(f"   Archive: {os.path.basename(archive_file)}")
        print(f"   Generated {len(refined_proposals)} code-based agent designs")

    def _save_task_agent_code(
        self, variant: TaskAgentVariant, tmp_dir: str, timestamp: str
    ):
        """Generate and save complete TaskAgent Python code for a variant"""

        # Generate the complete TaskAgent code
        code = self._generate_task_agent_code(variant)

        # Save to file
        code_file = os.path.join(
            tmp_dir, f"task_agent_{variant.variant_id}_{timestamp}.py"
        )
        with open(code_file, "w") as f:
            f.write(code)

    def _generate_task_agent_code(self, variant: TaskAgentVariant) -> str:
        """Generate complete Python code for a TaskAgent variant"""

        # Get system prompt based on variant parameters
        system_prompt = self._get_system_prompt_for_variant(variant)

        code = f'''"""
TaskAgent Variant: {variant.variant_id}
Generation: {variant.generation}
Parent: {variant.parent_id}
Mutation Type: {variant.mutation_type}
Created: {variant.created_at}

This is a complete, standalone TaskAgent implementation generated by the ADAS system.
"""

import asyncio
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# NOTE: API key should be provided via environment variable or config
# NEVER hardcode API keys in production code!

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    print("Warning: Cerebras SDK not available")
    Cerebras = None


@dataclass
class TaskAgentVariant:
    variant_id: str
    generation: int
    parent_id: Optional[str]
    mutation_type: str
    parameters: Dict[str, Any]
    performance_score: float = 0.0
    evaluation_count: int = 0
    created_at: str = ""


class TaskAgent:
    """
    TaskAgent Variant: {variant.variant_id}
    
    This agent responds to bottleneck situations with appropriate ATC actions.
    It analyzes aircraft states and provides ATC commands to resolve bottlenecks.
    
    Configuration:
    - Model: {variant.parameters.get('model', 'qwen-3-235b-a22b-thinking-2507')}
    - Temperature: {variant.parameters.get('temperature', 0.3)}
    - Max Tokens: {variant.parameters.get('max_tokens', 1000)}
    - Analysis Approach: {variant.parameters.get('analysis_approach', 'comprehensive')}
    - Prompt Style: {variant.parameters.get('prompt_style', 'structured')}
    """
    
    def __init__(self, cerebras_api_key: str):
        self.cerebras_client = None
        if Cerebras and cerebras_api_key:
            # Disable TCP warming to reduce API calls and avoid rate limiting
            self.cerebras_client = Cerebras(
                api_key=cerebras_api_key,
                warm_tcp_connection=False
            )
        
        # Variant configuration
        self.variant = TaskAgentVariant(
            variant_id="{variant.variant_id}",
            generation={variant.generation},
            parent_id="{variant.parent_id}",
            mutation_type="{variant.mutation_type}",
            parameters={json.dumps(variant.parameters, indent=12)},
            performance_score={variant.performance_score},
            evaluation_count={variant.evaluation_count},
            created_at="{variant.created_at}"
        )
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze bottleneck situation and provide ATC response"""
        
        aircraft_states = input_data.get("aircraft_states", {{}})
        bottleneck_info = input_data.get("bottleneck_info", {{}})
        
        # Build analysis prompt using variant's approach
        analysis_prompt = self._build_analysis_prompt(aircraft_states, bottleneck_info)
        
        if not self.cerebras_client:
            return {{
                "action": {{
                    "type": "no_action",
                    "command": "Unable to process - no API client",
                    "callsign": "N/A",
                    "priority": "low"
                }},
                "reasoning": "No AI client available",
                "confidence": 0.0,
            }}

        try:
            response = self.cerebras_client.chat.completions.create(
                messages=[{{"role": "user", "content": analysis_prompt}}],
                model="{variant.parameters.get('model', 'qwen-3-235b-a22b-thinking-2507')}",
                max_tokens={variant.parameters.get('max_tokens', 1000)},
                temperature={variant.parameters.get('temperature', 0.3)},
            )

            result_text = response.choices[0].message.content
            return self._parse_atc_response(result_text)

        except Exception as e:
            return {{
                "action": {{
                    "type": "error",
                    "command": f"Error processing request: {{e}}",
                    "callsign": "N/A",
                    "priority": "low"
                }},
                "reasoning": f"API Error: {{e}}",
                "confidence": 0.0,
            }}
    
    def _build_analysis_prompt(self, aircraft_states: Dict[str, Any], bottleneck_info: Dict[str, Any]) -> str:
        """Build analysis prompt based on variant's prompt style and approach"""
        
        # System prompt specific to this variant
        system_prompt = """{system_prompt}"""
        
        # Format aircraft data
        aircraft_data = json.dumps(aircraft_states, indent=2)
        bottleneck_data = json.dumps(bottleneck_info, indent=2)
        
        prompt = f"""{{system_prompt}}

AIRCRAFT STATES:
{{aircraft_data}}

BOTTLENECK INFORMATION:
{{bottleneck_data}}

Please analyze this situation and provide an appropriate ATC response following the format specified above."""
        
        return prompt
    
    def _parse_atc_response(self, response_text: str) -> Dict[str, Any]:
        """Parse ATC response from LLM output"""
        
        result = {{
            "action": {{
                "type": "unknown",
                "command": "Unable to parse response",
                "callsign": "N/A",
                "priority": "low"
            }},
            "reasoning": "Failed to parse LLM response",
            "confidence": 0.0,
        }}
        
        try:
            # Try to extract structured information from response
            lines = response_text.strip().split("\\n")
            
            for line in lines:
                line = line.strip()
                if line.startswith("Action Type:"):
                    result["action"]["type"] = line.split(":", 1)[1].strip()
                elif line.startswith("Command:"):
                    result["action"]["command"] = line.split(":", 1)[1].strip()
                elif line.startswith("Callsign:"):
                    result["action"]["callsign"] = line.split(":", 1)[1].strip()
                elif line.startswith("Priority:"):
                    result["action"]["priority"] = line.split(":", 1)[1].strip()
                elif line.startswith("Reasoning:"):
                    result["reasoning"] = line.split(":", 1)[1].strip()
                elif line.startswith("Confidence:"):
                    try:
                        confidence_str = line.split(":", 1)[1].strip().replace("%", "")
                        result["confidence"] = float(confidence_str) / 100.0
                    except:
                        result["confidence"] = 0.5
            
            # If we couldn't parse structured format, use the raw response
            if result["action"]["command"] == "Unable to parse response":
                result["action"]["command"] = response_text[:200] + "..." if len(response_text) > 200 else response_text
                result["reasoning"] = "Raw LLM response (unstructured)"
                result["confidence"] = 0.3
                
        except Exception as e:
            result["reasoning"] = f"Parse error: {{e}}"
        
        return result


# Example usage (for testing)
if __name__ == "__main__":
    import os
    
    # Get API key from environment
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        print("Please set CEREBRAS_API_KEY environment variable")
        exit(1)
    
    # Create TaskAgent instance
    agent = TaskAgent(api_key)
    
    # Example test data
    test_data = {{
        "aircraft_states": {{
            "AAL123": {{
                "callsign": "AAL123",
                "latitude": 40.6413,
                "longitude": -73.7781,
                "altitude": 3000,
                "speed": 250,
                "heading": 090,
                "aircraft_type": "B738"
            }},
            "UAL456": {{
                "callsign": "UAL456", 
                "latitude": 40.6500,
                "longitude": -73.7700,
                "altitude": 2800,
                "speed": 230,
                "heading": 270,
                "aircraft_type": "A320"
            }}
        }},
        "bottleneck_info": {{
            "type": "runway_congestion",
            "location": "KJFK_22L",
            "severity": "moderate",
            "estimated_delay": 180
        }}
    }}
    
    # Run async test
    async def test_agent():
        result = await agent.process(test_data)
        print("TaskAgent Response:")
        print(json.dumps(result, indent=2))
    
    # Run the test
    asyncio.run(test_agent())
'''

        return code

    def _get_system_prompt_for_variant(self, variant: TaskAgentVariant) -> str:
        """Get system prompt based on variant parameters"""

        prompt_style = variant.parameters.get("prompt_style", "structured")
        analysis_approach = variant.parameters.get("analysis_approach", "comprehensive")

        base_prompt = """You are an expert Air Traffic Controller AI assistant analyzing bottleneck situations at airports.

Your job is to analyze aircraft states and bottleneck information, then provide appropriate ATC commands to resolve the situation safely and efficiently."""

        if prompt_style == "structured":
            prompt = (
                base_prompt
                + """

ANALYSIS APPROACH: """
                + analysis_approach.upper()
                + """

Please provide your response in the following structured format:

Action Type: [clearance|vector|hold|speed_adjustment|altitude_change|runway_change|sequencing|other]
Command: [The exact ATC command to issue]
Callsign: [Aircraft callsign this command is for]
Priority: [high|medium|low]
Reasoning: [Brief explanation of why this action resolves the bottleneck]
Confidence: [Percentage confidence in this recommendation]

SAFETY REQUIREMENTS:
- Maintain minimum separation (3nm horizontal, 1000ft vertical)
- Consider aircraft performance limitations
- Prioritize safety over efficiency
- Use standard phraseology"""
            )

        elif prompt_style == "conversational":
            prompt = (
                base_prompt
                + """

Please analyze the situation naturally and provide your recommendation as an experienced controller would. Focus on """
                + analysis_approach
                + """ analysis.

Explain your reasoning and provide a clear ATC command. Consider:
- Aircraft separation and safety
- Efficient traffic flow
- Standard ATC procedures
- Airport-specific constraints"""
            )

        elif prompt_style == "technical":
            prompt = (
                base_prompt
                + """

Perform a """
                + analysis_approach
                + """ technical analysis considering:

TECHNICAL PARAMETERS:
- Aircraft performance envelopes
- Wake turbulence categories
- Runway capacity constraints
- Weather impact factors
- Navigation system capabilities

Provide technical justification for your recommended ATC action with precise parameters and expected outcomes."""
            )

        elif prompt_style == "safety_focused":
            prompt = (
                base_prompt
                + """

SAFETY-FIRST ANALYSIS using """
                + analysis_approach
                + """ approach:

MANDATORY SAFETY CHECKS:
- Minimum separation standards (3nm horizontal, 1000ft vertical)
- Wake turbulence considerations
- Weather impact assessment
- Emergency procedures availability
- Conflict resolution priorities

Prioritize safety over efficiency. When in doubt, choose the most conservative option.
Provide clear safety justification for all recommendations."""
            )

        elif prompt_style == "efficiency_focused":
            prompt = (
                base_prompt
                + """

EFFICIENCY-OPTIMIZED ANALYSIS using """
                + analysis_approach
                + """ approach:

OPTIMIZATION TARGETS:
- Minimize total delay time
- Maximize runway utilization
- Optimize aircraft sequencing
- Reduce fuel consumption
- Improve traffic flow rates

Balance efficiency gains with safety requirements. Focus on throughput optimization while maintaining safe operations."""
            )

        elif prompt_style == "hybrid":
            prompt = (
                base_prompt
                + """

HYBRID ANALYSIS combining multiple approaches with """
                + analysis_approach
                + """ methodology:

INTEGRATED CONSIDERATIONS:
- Safety requirements (non-negotiable minimums)
- Efficiency opportunities (where safe to optimize)
- Technical constraints (aircraft/airport limitations)
- Operational realities (controller workload, pilot experience)

Provide a balanced recommendation that considers all factors. Use structured format but with conversational explanations."""
            )

        else:  # default/other
            prompt = (
                base_prompt
                + """

Analyze this bottleneck situation using a """
                + analysis_approach
                + """ approach and provide an appropriate ATC response with clear reasoning."""
            )

        return prompt
