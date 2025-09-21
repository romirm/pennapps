"""
Simplified Training Script for GNN Bottleneck Prediction

This script provides a quick way to train your GNN model with minimal setup.
Run this to get started with training immediately.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import random
from datetime import datetime
from typing import Dict, List
import matplotlib.pyplot as plt
from tqdm import tqdm

from airport_bottleneck_model import AirportBottleneckModel
from config import BOTTLENECK_CONFIG
from adsb_processor import ADSBDataProcessor, AircraftData


def create_simple_training_data(num_samples: int = 100) -> List[Dict]:
    """Create simple synthetic training data"""
    print(f"Generating {num_samples} training samples...")

    training_data = []
    adsb_processor = ADSBDataProcessor(BOTTLENECK_CONFIG)

    for i in tqdm(range(num_samples)):
        # Create random scenario
        num_aircraft = random.randint(3, 15)
        bottleneck_severity = random.uniform(0.0, 1.0)

        # Generate aircraft data
        aircraft_list = []
        airport_lat, airport_lon = 40.6413, -73.7781  # JFK coordinates

        for j in range(num_aircraft):
            # Random positions around airport
            lat_offset = random.uniform(-0.01, 0.01)
            lon_offset = random.uniform(-0.01, 0.01)

            phase = random.choice(["approach", "departure", "taxi", "gate"])

            if phase == "approach":
                altitude = random.uniform(500, 2000)
                speed = random.uniform(120, 180)
            elif phase == "departure":
                altitude = random.uniform(0, 1000)
                speed = random.uniform(80, 150)
            elif phase == "taxi":
                altitude = random.uniform(0, 50)
                speed = random.uniform(5, 25)
            else:  # gate
                altitude = 0
                speed = 0

            aircraft = AircraftData(
                flight_id=f"SIM{j:03d}",
                aircraft_type=random.choice(["B737", "A320", "B777"]),
                latitude=airport_lat + lat_offset,
                longitude=airport_lon + lon_offset,
                altitude=altitude,
                heading=random.uniform(0, 360),
                speed=speed,
                phase=phase,
                timestamp=datetime.now().isoformat(),
            )
            aircraft_list.append(aircraft)

        # Build graph
        airport_config = {
            "icao": "KJFK",
            "runways": [{"id": "09L/27R"}],
            "gates": ["A1", "A2"],
            "taxiways": ["A", "B"],
        }

        graph_data = adsb_processor.construct_bottleneck_graph(
            aircraft_list, airport_config
        )

        # Simple labels based on scenario
        labels = {
            "bottleneck_probability": bottleneck_severity,
            "resolution_time": bottleneck_severity * 20 + random.uniform(-5, 5),
            "passengers_affected": num_aircraft * 150 * bottleneck_severity,
            "fuel_waste": num_aircraft * bottleneck_severity * 20,
        }

        training_data.append(
            {
                "graph_data": graph_data,
                "labels": labels,
                "num_aircraft": num_aircraft,
                "bottleneck_severity": bottleneck_severity,
            }
        )

    return training_data


def simple_train_loop():
    """Simple training loop"""
    print("🚀 Starting Simple GNN Training")
    print("=" * 50)

    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    print("🔧 Initializing model...")
    model = AirportBottleneckModel(BOTTLENECK_CONFIG)
    model.to(device)

    # Create training data
    print("📊 Creating training data...")
    training_data = create_simple_training_data(num_samples=50)

    # Split into train/val
    split_idx = int(0.8 * len(training_data))
    train_data = training_data[:split_idx]
    val_data = training_data[split_idx:]

    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}")

    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Training loop
    num_epochs = 20
    train_losses = []
    val_losses = []

    print("🎯 Starting training...")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_train_loss = 0

        for sample in train_data:
            optimizer.zero_grad()

            # Forward pass (simplified)
            graph_data = sample["graph_data"].to(device)

            try:
                # Get GNN predictions
                gnn_output = model.gnn(graph_data)

                # Simple prediction head
                bottleneck_pred = gnn_output["bottleneck_embeddings"].mean()

                # Simple loss
                target = torch.tensor(
                    sample["labels"]["bottleneck_probability"],
                    dtype=torch.float32,
                    device=device,
                )
                loss = criterion(bottleneck_pred, target)

                # Backward pass
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item()

            except Exception as e:
                print(f"Error in training step: {e}")
                continue

        # Validation phase
        model.eval()
        epoch_val_loss = 0

        with torch.no_grad():
            for sample in val_data:
                graph_data = sample["graph_data"].to(device)

                try:
                    gnn_output = model.gnn(graph_data)
                    bottleneck_pred = gnn_output["bottleneck_embeddings"].mean()

                    target = torch.tensor(
                        sample["labels"]["bottleneck_probability"],
                        dtype=torch.float32,
                        device=device,
                    )
                    loss = criterion(bottleneck_pred, target)

                    epoch_val_loss += loss.item()

                except Exception as e:
                    continue

        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(train_data)
        avg_val_loss = epoch_val_loss / len(val_data) if len(val_data) > 0 else 0

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}"
        )

    # Save model
    torch.save(model.state_dict(), "simple_trained_model.pth")
    print("✅ Model saved as 'simple_trained_model.pth'")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Simple Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig("simple_training_history.png")
    print("📊 Training plot saved as 'simple_training_history.png'")

    return model, train_losses, val_losses


def test_trained_model():
    """Test the trained model with new data"""
    print("\n🧪 Testing trained model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = AirportBottleneckModel(BOTTLENECK_CONFIG)
    model.load_state_dict(torch.load("simple_trained_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Create test data
    test_data = create_simple_training_data(num_samples=5)

    print("Test Results:")
    print("-" * 30)

    with torch.no_grad():
        for i, sample in enumerate(test_data):
            graph_data = sample["graph_data"].to(device)

            try:
                gnn_output = model.gnn(graph_data)
                prediction = gnn_output["bottleneck_embeddings"].mean().item()
                actual = sample["labels"]["bottleneck_probability"]

                print(f"Sample {i+1}:")
                print(f"  Predicted: {prediction:.3f}")
                print(f"  Actual: {actual:.3f}")
                print(f"  Error: {abs(prediction - actual):.3f}")
                print()

            except Exception as e:
                print(f"Error in test sample {i+1}: {e}")


if __name__ == "__main__":
    try:
        # Run simple training
        model, train_losses, val_losses = simple_train_loop()

        # Test the model
        test_trained_model()

        print("\n🎉 Simple training completed successfully!")
        print("Next steps:")
        print("1. Use the full training script (train_gnn.py) for better results")
        print("2. Integrate real ADS-B data for training")
        print("3. Fine-tune hyperparameters")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        print("This is expected if dependencies are missing.")
        print("Install requirements: pip install -r model/requirements.txt")

