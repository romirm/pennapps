#!/usr/bin/env python3
"""
Simple test script to verify GNN training works
"""

import torch
import sys
import os

# Add model directory to path
sys.path.append("model")

try:
    # Import modules directly
    import model.simple_training as simple_training

    print("✅ Successfully imported training modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def main():
    print("🧪 Testing Simple GNN Training")
    print("=" * 40)

    try:
        # Run simple training
        print("🚀 Starting simple training test...")
        model, train_losses, val_losses = simple_training.simple_train_loop()

        print("✅ Simple training completed!")
        print(f"📊 Final training loss: {train_losses[-1]:.4f}")
        print(f"📊 Final validation loss: {val_losses[-1]:.4f}")

        return True

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Test completed successfully!")
    else:
        print("\n💡 Try installing missing dependencies or check the error above")
