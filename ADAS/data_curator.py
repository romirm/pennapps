#!/usr/bin/env python3
"""
ATC Data Curator - Standalone Data Cleaning Tool

This script uses the CleaningAgent to intelligently select high-quality ATC training examples
from raw validation datasets and save them to the validation dataset folder.

Usage:
    python3.9 data_curator.py
    python3.9 data_curator.py --min-quality 0.7
    python3.9 data_curator.py --help
"""

import asyncio
import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

from cleaning_agent import CleaningAgent
from config import get_config


class DataCurator:
    """Standalone data curation tool for selecting high-quality ATC training examples"""

    def __init__(self, cerebras_api_key: str, base_path: str):
        self.cleaning_agent = CleaningAgent(cerebras_api_key)
        self.base_path = base_path
        self.raw_dir = os.path.join(base_path, "raw-validation-dataset")
        self.clean_dir = os.path.join(base_path, "validation-dataset")

        # Statistics tracking
        self.stats = {
            "total_processed": 0,
            "total_kept": 0,
            "files_processed": 0,
            "quality_distribution": {"excellent": 0, "good": 0, "fair": 0, "poor": 0},
            "top_examples": [],
        }

    async def curate_all_data(
        self, min_quality_score: float = 0.6, max_examples_to_show: int = 20
    ):
        """Curate all raw validation data files"""

        if not os.path.exists(self.raw_dir):
            print(f"❌ Raw validation directory not found: {self.raw_dir}")
            return

        os.makedirs(self.clean_dir, exist_ok=True)

        print("🎯 ATC Data Curator - Selecting High-Quality Training Examples")
        print(f"📂 Source: {self.raw_dir}")
        print(f"📁 Destination: {self.clean_dir}")
        print(f"🎚️  Minimum quality score: {min_quality_score}")
        print("=" * 70)

        # Process all JSON files in raw directory
        raw_files = [f for f in os.listdir(self.raw_dir) if f.endswith(".json")]

        if not raw_files:
            print("⚠️ No JSON files found in raw validation directory")
            return

        print(f"📋 Found {len(raw_files)} raw data files to process")

        for filename in raw_files:
            await self._process_file(filename, min_quality_score)

        # Generate final report
        await self._generate_final_report(max_examples_to_show)

    async def _process_file(self, filename: str, min_quality_score: float):
        """Process a single raw validation file"""

        raw_path = os.path.join(self.raw_dir, filename)

        try:
            print(f"\n📄 Processing: {filename}")

            with open(raw_path, "r") as f:
                raw_records = json.load(f)

            print(f"   📊 Total records: {len(raw_records)}")

            # Process each record
            high_quality_records = []
            file_stats = {"processed": 0, "kept": 0, "quality_scores": []}

            for i, record in enumerate(raw_records):
                # Progress indicator
                if i % 100 == 0 and i > 0:
                    print(f"   ⏳ Processed {i}/{len(raw_records)} records...")

                # Evaluate record quality
                cleaning_result = await self.cleaning_agent.process(
                    {"validation_record": record}
                )

                file_stats["processed"] += 1

                # Check if record meets quality threshold
                if cleaning_result.get("should_keep", False):
                    quality_score = cleaning_result.get("quality_score", 0.5)

                    if quality_score >= min_quality_score:
                        # Add quality metadata for tracking
                        enhanced_record = record.copy()
                        enhanced_record["_curation_metadata"] = {
                            "quality_score": quality_score,
                            "training_value": cleaning_result.get(
                                "training_value", "fair"
                            ),
                            "reasoning": cleaning_result.get("reasoning", ""),
                            "key_strengths": cleaning_result.get("key_strengths", []),
                            "issues_found": cleaning_result.get("issues_found", []),
                            "curated_at": datetime.now().isoformat(),
                            "source_file": filename,
                        }

                        high_quality_records.append(enhanced_record)
                        file_stats["kept"] += 1
                        file_stats["quality_scores"].append(quality_score)

                        # Track quality distribution
                        training_value = cleaning_result.get("training_value", "fair")
                        if training_value in self.stats["quality_distribution"]:
                            self.stats["quality_distribution"][training_value] += 1

                        # Track top examples
                        example_data = {
                            "source_file": filename,
                            "record_id": record.get("record_id", "unknown"),
                            "raw_transcription": record.get("atc_command", {}).get(
                                "raw_transcription", ""
                            ),
                            "quality_score": quality_score,
                            "training_value": training_value,
                            "reasoning": cleaning_result.get("reasoning", ""),
                        }
                        self.stats["top_examples"].append(example_data)

            # Save curated records
            if high_quality_records:
                # Sort by quality score (best first)
                high_quality_records.sort(
                    key=lambda x: x["_curation_metadata"]["quality_score"], reverse=True
                )

                # Remove metadata for clean training data
                clean_records = []
                for record in high_quality_records:
                    clean_record = record.copy()
                    del clean_record["_curation_metadata"]
                    clean_records.append(clean_record)

                # Save curated data
                output_filename = f"curated_{filename}"
                output_path = os.path.join(self.clean_dir, output_filename)

                with open(output_path, "w") as f:
                    json.dump(clean_records, f, indent=2)

                # Save curation report
                report_filename = f"curation_report_{filename}"
                report_path = os.path.join(self.clean_dir, report_filename)

                curation_report = {
                    "source_file": filename,
                    "curation_timestamp": datetime.now().isoformat(),
                    "statistics": {
                        "total_processed": file_stats["processed"],
                        "total_kept": file_stats["kept"],
                        "selection_rate": file_stats["kept"] / file_stats["processed"],
                        "average_quality": (
                            sum(file_stats["quality_scores"])
                            / len(file_stats["quality_scores"])
                            if file_stats["quality_scores"]
                            else 0
                        ),
                        "min_quality_threshold": min_quality_score,
                    },
                    "best_examples": [
                        {
                            "record_id": record.get("record_id", "unknown"),
                            "raw_transcription": record.get("atc_command", {}).get(
                                "raw_transcription", ""
                            ),
                            "quality_score": record["_curation_metadata"][
                                "quality_score"
                            ],
                            "training_value": record["_curation_metadata"][
                                "training_value"
                            ],
                            "reasoning": record["_curation_metadata"]["reasoning"],
                        }
                        for record in high_quality_records[:15]  # Top 15 examples
                    ],
                }

                with open(report_path, "w") as f:
                    json.dump(curation_report, f, indent=2)

                # Print file summary
                avg_quality = sum(file_stats["quality_scores"]) / len(
                    file_stats["quality_scores"]
                )
                print(
                    f"   ✅ Selected: {file_stats['kept']}/{file_stats['processed']} records ({file_stats['kept']/file_stats['processed']*100:.1f}%)"
                )
                print(f"   📈 Average quality: {avg_quality:.3f}")
                print(
                    f"   🏆 Best example: '{high_quality_records[0].get('atc_command', {}).get('raw_transcription', 'N/A')[:80]}...'"
                )
                print(f"   💾 Saved to: {output_filename}")
            else:
                print(f"   ⚠️ No records met quality threshold ({min_quality_score})")

            # Update global stats
            self.stats["total_processed"] += file_stats["processed"]
            self.stats["total_kept"] += file_stats["kept"]
            self.stats["files_processed"] += 1

        except Exception as e:
            print(f"   ❌ Error processing {filename}: {e}")

    async def _generate_final_report(self, max_examples_to_show: int):
        """Generate final curation report"""

        print("\n" + "=" * 70)
        print("🎯 DATA CURATION COMPLETE!")
        print("=" * 70)

        # Overall statistics
        selection_rate = (
            (self.stats["total_kept"] / self.stats["total_processed"] * 100)
            if self.stats["total_processed"] > 0
            else 0
        )

        print(f"📊 OVERALL STATISTICS:")
        print(f"   Files processed: {self.stats['files_processed']}")
        print(f"   Total records processed: {self.stats['total_processed']:,}")
        print(f"   High-quality records selected: {self.stats['total_kept']:,}")
        print(f"   Overall selection rate: {selection_rate:.1f}%")

        # Quality distribution
        print(f"\n📈 QUALITY DISTRIBUTION:")
        total_quality_records = sum(self.stats["quality_distribution"].values())
        for quality, count in self.stats["quality_distribution"].items():
            percentage = (
                (count / total_quality_records * 100)
                if total_quality_records > 0
                else 0
            )
            print(f"   {quality.capitalize()}: {count:,} records ({percentage:.1f}%)")

        # Top examples
        if self.stats["top_examples"]:
            # Sort all examples by quality score
            self.stats["top_examples"].sort(
                key=lambda x: x["quality_score"], reverse=True
            )

            print(
                f"\n🏆 TOP {min(max_examples_to_show, len(self.stats['top_examples']))} EXAMPLES:"
            )
            for i, example in enumerate(
                self.stats["top_examples"][:max_examples_to_show]
            ):
                print(
                    f"   {i+1:2d}. Quality: {example['quality_score']:.3f} | Value: {example['training_value']}"
                )
                print(
                    f"       Transcription: \"{example['raw_transcription'][:100]}{'...' if len(example['raw_transcription']) > 100 else ''}\""
                )
                print(
                    f"       Reasoning: {example['reasoning'][:120]}{'...' if len(example['reasoning']) > 120 else ''}"
                )
                print()

        # Save master curation report
        master_report_path = os.path.join(
            self.clean_dir,
            f"master_curation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        master_report = {
            "curation_timestamp": datetime.now().isoformat(),
            "statistics": self.stats,
            "configuration": {
                "source_directory": self.raw_dir,
                "output_directory": self.clean_dir,
                "total_files_found": self.stats["files_processed"],
            },
        }

        with open(master_report_path, "w") as f:
            json.dump(master_report, f, indent=2)

        print(f"📁 Curated datasets saved to: {self.clean_dir}")
        print(f"📋 Master report saved to: {os.path.basename(master_report_path)}")


async def main():
    """Main entry point for the data curator"""

    parser = argparse.ArgumentParser(
        description="ATC Data Curator - Select high-quality training examples from raw ATC data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3.9 data_curator.py                    # Use default quality threshold (0.6)
  python3.9 data_curator.py --min-quality 0.8 # Only keep highest quality examples
  python3.9 data_curator.py --show-examples 30 # Show top 30 examples in report
        """,
    )

    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.6,
        help="Minimum quality score threshold (0.0-1.0, default: 0.6)",
    )

    parser.add_argument(
        "--show-examples",
        type=int,
        default=20,
        help="Number of top examples to show in final report (default: 20)",
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help="Path to config file (default: use built-in config)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.min_quality <= 1.0:
        print("❌ Error: --min-quality must be between 0.0 and 1.0")
        sys.exit(1)

    if args.show_examples < 0:
        print("❌ Error: --show-examples must be non-negative")
        sys.exit(1)

    try:
        # Load configuration
        config = get_config()

        # Initialize curator
        curator = DataCurator(
            cerebras_api_key=config["cerebras_api_key"],
            base_path=config["paths"]["adas"],
        )

        # Run curation process
        await curator.curate_all_data(
            min_quality_score=args.min_quality, max_examples_to_show=args.show_examples
        )

    except KeyboardInterrupt:
        print("\n⏹️ Curation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error during curation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
