"""
ADAS System - Main Orchestrator

This is the main ADAS (Automated Design of Agentic Systems) orchestrator that
manages the evolutionary loop of task agent improvement using modular agents.
"""

import json
import os
import time
import random
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import asdict

from base_agent import TaskAgentVariant, EvaluationResult
from task_agent import TaskAgent
from meta_agent import MetaAgent
from evaluator_agent import EvaluatorAgent


class ADASystem:
    """
    Main ADAS (Automated Design of Agentic Systems) orchestrator
    Manages the evolutionary loop of task agent improvement
    """

    def __init__(self, cerebras_api_key: str, validation_dataset_path: str):
        self.cerebras_api_key = cerebras_api_key
        self.validation_dataset_path = validation_dataset_path

        # Initialize agents
        self.meta_agent = MetaAgent(cerebras_api_key)
        self.evaluator_agent = EvaluatorAgent(cerebras_api_key)

        # System state
        self.current_variants = []
        self.evaluation_history = []
        self.generation_count = 0

        # Create initial variant
        self._create_initial_variant()

    def _create_initial_variant(self):
        """Create the first task agent variant"""
        from config import get_config

        config = get_config()

        initial_variant = TaskAgentVariant(
            variant_id="gen0_baseline",
            generation=0,
            parent_id=None,
            mutation_type="baseline",
            parameters=config["task_agent"]["baseline_parameters"].copy(),
            created_at=datetime.now().isoformat(),
        )
        self.current_variants.append(initial_variant)

    async def run_evolution_cycle(
        self, num_cycles: int = 5, evaluations_per_variant: int = 10
    ) -> Dict[str, Any]:
        """Run the main ADAS evolution cycle"""

        results = {
            "cycles_completed": 0,
            "best_variant": None,
            "performance_history": [],
            "generation_summary": [],
        }

        for cycle in range(num_cycles):
            print(f"\n🔄 Starting Evolution Cycle {cycle + 1}/{num_cycles}")

            # Step 1: Evaluate current variants
            await self._evaluate_variants(evaluations_per_variant)

            # Step 2: Generate new variants via meta-agent
            new_variants = await self._generate_new_variants()

            # Step 3: Update variant pool
            self._update_variant_pool(new_variants)

            # Step 4: Track progress
            cycle_summary = self._summarize_cycle(cycle + 1)
            results["generation_summary"].append(cycle_summary)

            results["cycles_completed"] = cycle + 1

            print(f"✅ Completed Cycle {cycle + 1}")
            print(f"   Current variants: {len(self.current_variants)}")
            print(f"   Best performance: {cycle_summary['best_score']:.3f}")

        # Find best overall variant
        best_variant = max(self.current_variants, key=lambda v: v.performance_score)
        results["best_variant"] = best_variant
        results["performance_history"] = self.evaluation_history

        return results

    async def _evaluate_variants(self, num_evaluations: int):
        """Evaluate all current variants on validation data"""

        validation_records = self._load_validation_data()

        if not validation_records:
            print("⚠️ No validation data found")
            return

        for variant in self.current_variants:
            print(f"📊 Evaluating variant {variant.variant_id}")

            task_agent = TaskAgent(self.cerebras_api_key, variant)

            # Sample validation records for evaluation
            sampled_records = random.sample(
                validation_records, min(num_evaluations, len(validation_records))
            )

            scores = []

            for record in sampled_records:
                try:
                    # Ensure record is a dictionary
                    if isinstance(record, str):
                        print(
                            f"⚠️ Warning: Record is a string, skipping: {record[:100]}..."
                        )
                        continue

                    if not isinstance(record, dict):
                        print(f"⚠️ Warning: Record is not a dict, type: {type(record)}")
                        continue

                    # Task agent predicts action
                    prediction = await task_agent.process(
                        {"aircraft_states": record.get("aircraft_states", {})}
                    )

                    # Evaluator judges performance
                    evaluation = await self.evaluator_agent.process(
                        {
                            "predicted_action": prediction.get("predicted_action", {}),
                            "expected_action": record.get("atc_command", {}),
                            "aircraft_states": record.get("aircraft_states", {}),
                        }
                    )

                    score = evaluation.get("score", 0.0)
                    scores.append(score)

                    # Record detailed evaluation
                    eval_result = EvaluationResult(
                        variant_id=variant.variant_id,
                        score=score,
                        reasoning=evaluation.get("reasoning", ""),
                        validation_record_id=record.get("record_id", ""),
                        timestamp=datetime.now().isoformat(),
                    )

                    self.evaluation_history.append(eval_result)

                except Exception as e:
                    print(f"❌ Error evaluating variant {variant.variant_id}: {e}")
                    scores.append(0.0)

            # Update variant performance
            if scores:
                variant.performance_score = sum(scores) / len(scores)
                variant.evaluation_count += len(scores)

                print(
                    f"   Score: {variant.performance_score:.3f} ({len(scores)} evaluations)"
                )

    async def _generate_new_variants(self) -> List[TaskAgentVariant]:
        """Generate new variants using meta-agent"""

        print("🧠 Meta-agent generating new variants...")

        # Get recent performance data
        recent_evaluations = [
            e for e in self.evaluation_history[-100:]
        ]  # Last 100 evaluations

        mutation_result = await self.meta_agent.process(
            {
                "current_variants": self.current_variants,
                "performance_data": recent_evaluations,
            }
        )

        new_variants = mutation_result.get("variants", [])

        print(f"   Generated {len(new_variants)} new variants")

        return new_variants

    def _update_variant_pool(self, new_variants: List[TaskAgentVariant]):
        """Update the pool of variants, keeping best performers with diversity"""

        # Always preserve the current best variant (parent option)
        best_current = None
        if self.current_variants:
            best_current = max(self.current_variants, key=lambda v: v.performance_score)
            print(
                f"🏆 Preserving best parent variant: {best_current.variant_id} (score: {best_current.performance_score:.3f})"
            )

        # Add new variants
        self.current_variants.extend(new_variants)

        # Sort by performance and keep top variants
        self.current_variants.sort(key=lambda v: v.performance_score, reverse=True)

        # Keep top variants with diversity consideration
        from config import get_config

        config = get_config()
        max_variants = config["evolution"].get("max_variants", 5)

        if len(self.current_variants) > max_variants:
            kept_variants = self._select_diverse_variants(
                self.current_variants, max_variants, best_current
            )
            removed = [v for v in self.current_variants if v not in kept_variants]
            self.current_variants = kept_variants

            print(
                f"   Kept {len(kept_variants)} variants, removed {len(removed)} lower-performing variants"
            )
            if best_current:
                print(f"   ✅ Best parent variant preserved in pool")

    def _select_diverse_variants(
        self,
        variants: List[TaskAgentVariant],
        max_count: int,
        best_current: TaskAgentVariant = None,
    ) -> List[TaskAgentVariant]:
        """Select variants balancing performance and diversity"""

        if len(variants) <= max_count:
            return variants

        # Always include best performer
        selected = [variants[0]]
        remaining = variants[1:]

        # Ensure best parent is included if it exists
        if best_current and best_current not in selected:
            if best_current in remaining:
                selected.append(best_current)
                remaining.remove(best_current)

        # Select remaining variants balancing performance and diversity
        while len(selected) < max_count and remaining:
            # Score variants based on performance and diversity
            scored_variants = []
            for variant in remaining:
                performance_score = variant.performance_score

                # Diversity bonus - prefer different mutation types
                diversity_bonus = (
                    0.1
                    if variant.mutation_type not in [v.mutation_type for v in selected]
                    else 0
                )

                # Generation bonus - slightly prefer newer variants
                generation_bonus = variant.generation * 0.01

                total_score = performance_score + diversity_bonus + generation_bonus
                scored_variants.append((total_score, variant))

            # Select best scoring variant (sort by score, which is first element of tuple)
            scored_variants.sort(key=lambda x: x[0], reverse=True)
            best_variant = scored_variants[0][1]
            selected.append(best_variant)
            remaining.remove(best_variant)

        return selected

    def _summarize_cycle(self, cycle_num: int) -> Dict[str, Any]:
        """Summarize performance of current cycle"""

        if not self.current_variants:
            return {"cycle": cycle_num, "best_score": 0.0, "avg_score": 0.0}

        scores = [
            v.performance_score for v in self.current_variants if v.evaluation_count > 0
        ]

        return {
            "cycle": cycle_num,
            "best_score": max(scores) if scores else 0.0,
            "avg_score": sum(scores) / len(scores) if scores else 0.0,
            "variants_count": len(self.current_variants),
            "total_evaluations": sum(v.evaluation_count for v in self.current_variants),
        }

    def _load_validation_data(self) -> List[Dict[str, Any]]:
        """Load validation data from dataset files"""

        validation_records = []

        try:
            # Load from validation-dataset folder (curated ATC transcripts)
            dataset_dir = os.path.join(
                self.validation_dataset_path, "validation-dataset"
            )

            curated_count = 0
            if os.path.exists(dataset_dir):
                for filename in os.listdir(dataset_dir):
                    # Only load curated data files, not reports
                    if filename.endswith(".json") and filename.startswith("curated_"):
                        filepath = os.path.join(dataset_dir, filename)

                        try:
                            with open(filepath, "r") as f:
                                records = json.load(f)
                                if isinstance(records, list):
                                    # Filter out any non-dict items
                                    valid_records = [
                                        r for r in records if isinstance(r, dict)
                                    ]
                                    validation_records.extend(valid_records)
                                    curated_count += len(valid_records)
                                    if len(valid_records) != len(records):
                                        print(
                                            f"⚠️ Filtered out {len(records) - len(valid_records)} invalid records from {filename}"
                                        )
                                else:
                                    print(
                                        f"⚠️ Skipping {filename}: not a list of records"
                                    )
                        except Exception as e:
                            print(f"❌ Error loading {filename}: {e}")

            # Load from bottleneck-scenarios folder (synthetic bottleneck scenarios)
            bottleneck_dir = os.path.join(
                self.validation_dataset_path, "bottleneck-scenarios"
            )

            bottleneck_count = 0
            if os.path.exists(bottleneck_dir):
                for filename in os.listdir(bottleneck_dir):
                    if filename.endswith(".json") and filename.startswith(
                        "bottleneck_scenarios"
                    ):
                        filepath = os.path.join(bottleneck_dir, filename)

                        try:
                            with open(filepath, "r") as f:
                                records = json.load(f)
                                if isinstance(records, list):
                                    valid_records = [
                                        r for r in records if isinstance(r, dict)
                                    ]
                                    # Transform bottleneck scenarios to match validation format
                                    transformed_records = []
                                    for record in valid_records:
                                        transformed = {
                                            "record_id": record.get("record_id"),
                                            "aircraft_states": record.get(
                                                "aircraft_states", {}
                                            ),
                                            "atc_command": record.get(
                                                "expected_atc_action", {}
                                            ),
                                            "bottleneck_info": record.get(
                                                "bottleneck_info", {}
                                            ),
                                            "scenario_type": record.get(
                                                "scenario_type", "unknown"
                                            ),
                                            "training_notes": record.get(
                                                "training_notes", ""
                                            ),
                                        }
                                        transformed_records.append(transformed)

                                    validation_records.extend(transformed_records)
                                    bottleneck_count += len(transformed_records)
                        except Exception as e:
                            print(
                                f"❌ Error loading bottleneck scenarios {filename}: {e}"
                            )

            # Load from taxi-hold-scenarios folder (taxi and hold scenarios)
            taxi_hold_dir = os.path.join(
                self.validation_dataset_path, "taxi-hold-scenarios"
            )

            taxi_hold_count = 0
            if os.path.exists(taxi_hold_dir):
                for filename in os.listdir(taxi_hold_dir):
                    if filename.endswith(".json") and filename.startswith(
                        "taxi_hold_scenarios"
                    ):
                        filepath = os.path.join(taxi_hold_dir, filename)

                        try:
                            with open(filepath, "r") as f:
                                records = json.load(f)
                                if isinstance(records, list):
                                    valid_records = [
                                        r for r in records if isinstance(r, dict)
                                    ]
                                    # Transform taxi/hold scenarios to match validation format
                                    transformed_records = []
                                    for record in valid_records:
                                        transformed = {
                                            "record_id": record.get("record_id"),
                                            "aircraft_states": record.get(
                                                "aircraft_states", {}
                                            ),
                                            "atc_command": record.get(
                                                "expected_atc_action", {}
                                            ),
                                            "bottleneck_info": record.get(
                                                "bottleneck_info", {}
                                            ),
                                            "scenario_type": record.get(
                                                "scenario_type", "unknown"
                                            ),
                                            "training_notes": record.get(
                                                "training_notes", ""
                                            ),
                                        }
                                        transformed_records.append(transformed)

                                    validation_records.extend(transformed_records)
                                    taxi_hold_count += len(transformed_records)
                        except Exception as e:
                            print(
                                f"❌ Error loading taxi/hold scenarios {filename}: {e}"
                            )

            print(f"📚 Loaded {len(validation_records)} validation records:")
            print(f"   📝 Curated transcripts: {curated_count}")
            print(f"   🏭 Bottleneck scenarios: {bottleneck_count}")
            print(f"   🚕 Taxi & hold scenarios: {taxi_hold_count}")

        except Exception as e:
            print(f"❌ Error loading validation data: {e}")

        return validation_records

    def save_system_state(self, filepath: str):
        """Save current system state to file"""

        state = {
            "current_variants": [asdict(v) for v in self.current_variants],
            "evaluation_history": [asdict(e) for e in self.evaluation_history],
            "generation_count": self.generation_count,
            "timestamp": datetime.now().isoformat(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        print(f"💾 System state saved to {filepath}")

    def load_system_state(self, filepath: str):
        """Load system state from file"""

        try:
            with open(filepath, "r") as f:
                state = json.load(f)

            # Reconstruct variants
            self.current_variants = [
                TaskAgentVariant(**v) for v in state.get("current_variants", [])
            ]

            # Reconstruct evaluation history
            self.evaluation_history = [
                EvaluationResult(**e) for e in state.get("evaluation_history", [])
            ]

            self.generation_count = state.get("generation_count", 0)

            print(f"📂 System state loaded from {filepath}")
            print(f"   Variants: {len(self.current_variants)}")
            print(f"   Evaluations: {len(self.evaluation_history)}")

        except Exception as e:
            print(f"❌ Error loading system state: {e}")


# Example usage and testing functions
async def main():
    """Example usage of the ADAS system"""

    # Get configuration
    from config import get_config

    config = get_config()

    # Initialize system
    adas = ADASystem(
        cerebras_api_key=config["cerebras_api_key"],
        validation_dataset_path=config["paths"]["adas"],
    )

    # Note: Use data_curator.py to clean raw validation data first if needed

    # Run evolution cycles
    results = await adas.run_evolution_cycle(
        num_cycles=config["evolution"]["num_cycles"],
        evaluations_per_variant=config["evolution"]["evaluations_per_variant"],
    )

    print("\n🎯 ADAS Evolution Complete!")
    print(f"Cycles completed: {results['cycles_completed']}")
    print(f"Best variant: {results['best_variant'].variant_id}")
    print(f"Best score: {results['best_variant'].performance_score:.3f}")

    return results


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
