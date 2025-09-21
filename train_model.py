#!/usr/bin/env python3
"""
Main Training Script for GNN-KAN Airport Bottleneck Prediction System

Usage:
    python train_model.py --config development  # Quick training for testing
    python train_model.py --config production   # Full training
    python train_model.py --resume checkpoint.pth  # Resume from checkpoint
"""

import argparse
import os
import sys
import torch
import json
from datetime import datetime

# Add model directory to path
sys.path.append("model")

try:
    from model.train_gnn import BottleneckDataset, BottleneckTrainer
    from model.airport_bottleneck_model import AirportBottleneckModel
    from model.config import BOTTLENECK_CONFIG
    from model.training_config import TRAINING_CONFIG, QUICK_START_CONFIGS
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the project root directory")
    sys.exit(1)


def setup_training_environment():
    """Setup training environment and check dependencies"""
    print("🔧 Setting up training environment...")

    # Check PyTorch installation
    print(f"PyTorch version: {torch.__version__}")

    # Check device availability
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA available - GPU: {torch.cuda.get_device_name(0)}")
        print(
            f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
        )
    else:
        device = torch.device("cpu")
        print("⚠️  CUDA not available - using CPU (training will be slower)")

    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("plots", exist_ok=True)

    return device


def create_training_config(config_name: str = "development") -> dict:
    """Create training configuration"""
    if config_name in QUICK_START_CONFIGS:
        config = QUICK_START_CONFIGS[config_name].copy()
        print(f"📋 Using '{config_name}' configuration")
    else:
        config = TRAINING_CONFIG.copy()
        print(f"📋 Using default training configuration")

    # Merge with base bottleneck config
    merged_config = {**BOTTLENECK_CONFIG, **config}

    return merged_config


def train_model(config: dict, resume_checkpoint: str = None):
    """Main training function"""

    print("🚀 Starting GNN-KAN Bottleneck Prediction Training")
    print("=" * 60)

    # Print configuration summary
    print("📋 Training Configuration:")
    print(f"   Train samples: {config['train_samples']}")
    print(f"   Val samples: {config['val_samples']}")
    print(f"   Epochs: {config['num_epochs']}")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Learning rate: {config['learning_rate']}")
    print(f"   Airports: {config['airports']}")

    # Create datasets
    print("\n📊 Creating datasets...")
    try:
        train_dataset = BottleneckDataset(
            num_samples=config["train_samples"], airports=config["airports"]
        )
        print(f"✅ Training dataset created: {len(train_dataset)} samples")

        val_dataset = BottleneckDataset(
            num_samples=config["val_samples"],
            airports=config["airports"][:2],  # Use subset for validation
        )
        print(f"✅ Validation dataset created: {len(val_dataset)} samples")

    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return None

    # Initialize model
    print("\n🔧 Initializing model...")
    try:
        model = AirportBottleneckModel(config)
        print(f"✅ Model initialized")
        print(f"   GNN layers: {config['gnn_layers']}")
        print(f"   Hidden dim: {config['gnn_hidden_dim']}")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"❌ Error initializing model: {e}")
        return None

    # Create trainer
    print("\n🎯 Setting up trainer...")
    try:
        trainer = BottleneckTrainer(model, config)
        print("✅ Trainer initialized")

        # Resume from checkpoint if specified
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            print(f"📂 Resuming from checkpoint: {resume_checkpoint}")
            trainer.load_model(resume_checkpoint)

    except Exception as e:
        print(f"❌ Error setting up trainer: {e}")
        return None

    # Start training
    print("\n🚀 Starting training...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        history = trainer.train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            num_epochs=config["num_epochs"],
            batch_size=config["batch_size"],
        )

        print("✅ Training completed successfully!")

        # Save training history
        history_file = f"logs/training_history_{timestamp}.json"
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)
        print(f"📊 Training history saved: {history_file}")

        # Plot training history
        try:
            trainer.plot_training_history()
            print("📈 Training plots saved")
        except Exception as e:
            print(f"⚠️  Could not save plots: {e}")

        # Save final model with timestamp
        final_model_path = f"models/bottleneck_model_{timestamp}.pth"
        trainer.save_model(final_model_path)
        print(f"💾 Final model saved: {final_model_path}")

        return trainer, history

    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None


def evaluate_model(model_path: str, config: dict):
    """Evaluate trained model"""
    print(f"\n🧪 Evaluating model: {model_path}")

    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return

    try:
        # Load model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AirportBottleneckModel(config)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        print("✅ Model loaded successfully")

        # Create test dataset
        test_dataset = BottleneckDataset(num_samples=50, airports=["KJFK", "KLAX"])

        # Run evaluation
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for i, sample in enumerate(test_dataset):
                if i >= 10:  # Test on first 10 samples
                    break

                graph_data = sample["graph_data"].to(device)
                gnn_output = model.gnn(graph_data)

                # Simple evaluation metric
                predicted_bottleneck = gnn_output["bottleneck_embeddings"].mean().item()
                actual_severity = sample["metadata"]["scenario_params"][
                    "bottleneck_severity"
                ]

                # Check if prediction is within reasonable range
                if abs(predicted_bottleneck - actual_severity) < 0.3:
                    correct_predictions += 1
                total_predictions += 1

                print(
                    f"Sample {i+1}: Predicted={predicted_bottleneck:.3f}, Actual={actual_severity:.3f}"
                )

        accuracy = (
            correct_predictions / total_predictions if total_predictions > 0 else 0
        )
        print(f"\n📊 Evaluation Results:")
        print(f"   Accuracy: {accuracy:.2%}")
        print(f"   Correct predictions: {correct_predictions}/{total_predictions}")

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Train GNN-KAN Bottleneck Prediction Model"
    )
    parser.add_argument(
        "--config",
        default="development",
        choices=["development", "testing", "production"],
        help="Training configuration preset",
    )
    parser.add_argument("--resume", type=str, help="Resume from checkpoint file")
    parser.add_argument(
        "--evaluate", type=str, help="Evaluate model from checkpoint file"
    )
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, help="Override batch size")

    args = parser.parse_args()

    # Setup environment
    device = setup_training_environment()

    # Create configuration
    config = create_training_config(args.config)

    # Override config with command line arguments
    if args.epochs:
        config["num_epochs"] = args.epochs
        print(f"🔧 Overriding epochs: {args.epochs}")

    if args.batch_size:
        config["batch_size"] = args.batch_size
        print(f"🔧 Overriding batch size: {args.batch_size}")

    # Run evaluation if requested
    if args.evaluate:
        evaluate_model(args.evaluate, config)
        return

    # Run training
    result = train_model(config, args.resume)

    if result:
        trainer, history = result
        print(f"\n🎉 Training completed successfully!")
        print(f"📈 Final training loss: {history['loss'][-1]:.4f}")
        print(f"📈 Final validation loss: {history['val_loss'][-1]:.4f}")

        # Suggest next steps
        print(f"\n📋 Next Steps:")
        print(
            f"1. Evaluate model: python train_model.py --evaluate models/best_bottleneck_model.pth"
        )
        print(
            f"2. Test with real data: python model/test_bottleneck_prediction.py --sample"
        )
        print(f"3. Deploy model: python flight_bottleneck_predictor.py")

    else:
        print(f"\n❌ Training failed. Check the error messages above.")
        print(f"💡 Try running with --config testing for a smaller test run")


if __name__ == "__main__":
    main()

