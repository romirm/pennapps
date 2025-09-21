#!/usr/bin/env python3
"""
ADAS Runner Script
Easy-to-use interface for running the Automated Design of Agentic Systems
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adas_system import ADASystem
from config import get_config, validate_config


def setup_logging():
    """Setup logging configuration"""
    import logging

    config = get_config()
    logging_config = config["logging"]

    logging.basicConfig(
        level=getattr(logging, logging_config["log_level"]),
        format=logging_config["log_format"],
        handlers=[
            logging.FileHandler(logging_config["log_file"]),
            logging.StreamHandler(),
        ],
    )

    return logging.getLogger(__name__)


async def run_full_evolution(args):
    """Run complete ADAS evolution cycle"""

    logger = setup_logging()
    config = get_config()

    logger.info("🚀 Starting ADAS Full Evolution")
    logger.info(f"Cycles: {args.cycles}")
    logger.info(f"Evaluations per variant: {args.evaluations}")

    # Initialize system
    adas = ADASystem(
        cerebras_api_key=config["cerebras_api_key"],
        validation_dataset_path=config["paths"]["validation_dataset"],
    )

    # Clean raw data if requested
    if args.clean_data:
        logger.warning("⚠️ Data cleaning has been moved to a separate tool!")
        logger.info("🧹 Please run: python3.9 data_curator.py")
        logger.info("   This will intelligently select high-quality training examples.")
        return

    # Run evolution
    results = await adas.run_evolution_cycle(
        num_cycles=args.cycles, evaluations_per_variant=args.evaluations
    )

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        config["paths"]["adas"], f"adas_results_{timestamp}.json"
    )

    # Prepare results for JSON serialization
    json_results = {
        "cycles_completed": results["cycles_completed"],
        "best_variant": (
            results["best_variant"].__dict__ if results["best_variant"] else None
        ),
        "generation_summary": results["generation_summary"],
        "total_evaluations": len(results["performance_history"]),
        "configuration": config,
        "timestamp": timestamp,
    }

    with open(results_file, "w") as f:
        json.dump(json_results, f, indent=2)

    logger.info(f"💾 Results saved to: {results_file}")

    # Display summary
    print("\n" + "=" * 60)
    print("🏆 ADAS EVOLUTION COMPLETE")
    print("=" * 60)

    if results["best_variant"]:
        best = results["best_variant"]
        print(f"Best Variant: {best.variant_id}")
        print(f"Performance Score: {best.performance_score:.3f}")
        print(f"Generation: {best.generation}")
        print(f"Mutation Type: {best.mutation_type}")
        print(f"Evaluations: {best.evaluation_count}")

    print(f"\nTotal Cycles: {results['cycles_completed']}")
    print(f"Total Evaluations: {len(results['performance_history'])}")

    return results


async def run_single_evaluation(args):
    """Run single evaluation test"""

    logger = setup_logging()
    config = get_config()

    logger.info("🧪 Running Single Evaluation Test")

    adas = ADASystem(
        cerebras_api_key=config["cerebras_api_key"],
        validation_dataset_path=config["paths"]["validation_dataset"],
    )

    # Load validation data
    validation_records = adas._load_validation_data()

    if not validation_records:
        print("❌ No validation data found")
        return

    # Use first validation record for testing
    test_record = validation_records[0]

    # Handle case where record might be a string
    if isinstance(test_record, str):
        print(f"⚠️ Record is a string, skipping: {test_record[:100]}...")
        return

    if not isinstance(test_record, dict):
        print(f"⚠️ Record is not a dict, type: {type(test_record)}")
        return

    print(f"🔍 Testing with record: {test_record.get('record_id', 'unknown')}")

    # Get baseline variant
    baseline_variant = adas.current_variants[0]

    from adas_system import TaskAgent

    task_agent = TaskAgent(config["cerebras_api_key"], baseline_variant)

    # Run prediction
    prediction = await task_agent.process(
        {"aircraft_states": test_record.get("aircraft_states", {})}
    )

    # Run evaluation
    evaluation = await adas.evaluator_agent.process(
        {
            "predicted_action": prediction.get("predicted_action", {}),
            "expected_action": test_record.get("atc_command", {}),
            "aircraft_states": test_record.get("aircraft_states", {}),
        }
    )

    # Display results
    print("\n" + "=" * 50)
    print("📊 SINGLE EVALUATION RESULTS")
    print("=" * 50)

    print(f"\nPredicted Action:")
    print(json.dumps(prediction.get("predicted_action", {}), indent=2))

    print(f"\nExpected Action:")
    print(json.dumps(test_record.get("atc_command", {}), indent=2))

    print(f"\nEvaluation Score: {evaluation.get('score', 0.0):.3f}")
    print(f"Reasoning: {evaluation.get('reasoning', 'No reasoning provided')}")
    print(f"Safety Assessment: {evaluation.get('safety_assessment', 'Unknown')}")


async def clean_validation_data(args):
    """Clean raw validation data - now redirects to standalone curator"""

    print("⚠️ Data cleaning has been moved to a separate, more powerful tool!")
    print("")
    print("🧹 To curate high-quality training data, run:")
    print("   python3.9 data_curator.py")
    print("")
    print("📋 Available options:")
    print("   python3.9 data_curator.py --min-quality 0.7  # Higher quality threshold")
    print("   python3.9 data_curator.py --show-examples 30  # Show more examples")
    print("   python3.9 data_curator.py --help              # See all options")
    print("")
    print("✨ The new curator provides:")
    print("   • Intelligent quality assessment")
    print("   • No limits on good datasets (keeps all qualifying records)")
    print("   • Detailed curation reports")
    print("   • Quality scoring and ranking")


def analyze_results(args):
    """Analyze previous ADAS results"""

    if not os.path.exists(args.results_file):
        print(f"❌ Results file not found: {args.results_file}")
        return

    with open(args.results_file, "r") as f:
        results = json.load(f)

    print("📊 ADAS RESULTS ANALYSIS")
    print("=" * 40)

    print(f"Timestamp: {results.get('timestamp', 'Unknown')}")
    print(f"Cycles Completed: {results.get('cycles_completed', 0)}")
    print(f"Total Evaluations: {results.get('total_evaluations', 0)}")

    if results.get("best_variant"):
        best = results["best_variant"]
        print(f"\n🏆 Best Variant:")
        print(f"  ID: {best.get('variant_id', 'Unknown')}")
        print(f"  Score: {best.get('performance_score', 0.0):.3f}")
        print(f"  Generation: {best.get('generation', 0)}")
        print(f"  Mutation: {best.get('mutation_type', 'Unknown')}")

    # Performance trend
    generation_summary = results.get("generation_summary", [])
    if generation_summary:
        print(f"\n📈 Performance Trend:")
        for summary in generation_summary:
            cycle = summary.get("cycle", 0)
            best_score = summary.get("best_score", 0.0)
            avg_score = summary.get("avg_score", 0.0)
            print(f"  Cycle {cycle}: Best={best_score:.3f}, Avg={avg_score:.3f}")


def main():
    """Main CLI interface"""

    parser = argparse.ArgumentParser(
        description="ADAS - Automated Design of Agentic Systems"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Full evolution command
    evolve_parser = subparsers.add_parser("evolve", help="Run full evolution cycle")
    evolve_parser.add_argument(
        "--cycles", type=int, default=5, help="Number of evolution cycles"
    )
    evolve_parser.add_argument(
        "--evaluations", type=int, default=10, help="Evaluations per variant"
    )
    evolve_parser.add_argument(
        "--clean-data", action="store_true", help="Clean raw data before evolution"
    )

    # Single evaluation command
    test_parser = subparsers.add_parser("test", help="Run single evaluation test")

    # Data cleaning command
    clean_parser = subparsers.add_parser("clean", help="Clean raw validation data")

    # Results analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze previous results")
    analyze_parser.add_argument("results_file", help="Path to results JSON file")

    # Configuration validation
    config_parser = subparsers.add_parser("config", help="Validate configuration")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Validate configuration
    if not validate_config():
        print("❌ Configuration validation failed")
        sys.exit(1)

    # Run command
    if args.command == "evolve":
        asyncio.run(run_full_evolution(args))
    elif args.command == "test":
        asyncio.run(run_single_evaluation(args))
    elif args.command == "clean":
        asyncio.run(clean_validation_data(args))
    elif args.command == "analyze":
        analyze_results(args)
    elif args.command == "config":
        print("✅ Configuration is valid")
        config = get_config()
        print("\nCurrent configuration:")
        print(json.dumps(config, indent=2, default=str))


if __name__ == "__main__":
    main()
