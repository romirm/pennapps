#!/usr/bin/env python3
"""
Quick test of GNN training without complex imports
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from datetime import datetime
import sys
import os

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_basic_gnn():
    """Test basic GNN functionality"""
    print("🧪 Testing Basic GNN Components")
    print("=" * 40)

    try:
        # Test PyTorch Geometric
        from torch_geometric.nn import MessagePassing
        from torch_geometric.data import Data, Batch

        print("✅ PyTorch Geometric imported successfully")

        # Create simple test data
        x = torch.randn(10, 5)  # 10 nodes, 5 features each
        edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
        data = Data(x=x, edge_index=edge_index)
        print("✅ Graph data created successfully")

        # Simple GNN layer test
        class SimpleGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin1 = nn.Linear(5, 32)
                self.lin2 = nn.Linear(32, 16)
                self.lin3 = nn.Linear(16, 1)

            def forward(self, x):
                x = torch.relu(self.lin1(x))
                x = torch.relu(self.lin2(x))
                return torch.sigmoid(self.lin3(x))

        model = SimpleGNN()
        print("✅ Simple GNN model created")

        # Test forward pass
        output = model(data.x)
        print(f"✅ Forward pass successful, output shape: {output.shape}")

        # Test training step
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()

        target = torch.randint(0, 2, (10, 1)).float()
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"✅ Training step successful, loss: {loss.item():.4f}")

        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_synthetic_data():
    """Test synthetic data generation"""
    print("\n🔧 Testing Synthetic Data Generation")
    print("-" * 40)

    try:
        # Create synthetic airport scenario
        num_aircraft = 5
        aircraft_data = []

        for i in range(num_aircraft):
            aircraft = {
                "id": f"SIM{i:03d}",
                "type": random.choice(["B737", "A320", "B777"]),
                "lat": 40.6413 + random.uniform(-0.01, 0.01),
                "lon": -73.7781 + random.uniform(-0.01, 0.01),
                "alt": random.uniform(0, 2000),
                "speed": random.uniform(0, 200),
                "phase": random.choice(["approach", "departure", "taxi", "gate"]),
            }
            aircraft_data.append(aircraft)

        print(f"✅ Created {len(aircraft_data)} synthetic aircraft")

        # Create graph from aircraft data
        node_features = []
        for aircraft in aircraft_data:
            features = [
                aircraft["alt"] / 1000.0,
                aircraft["speed"] / 100.0,
                1.0 if aircraft["phase"] == "approach" else 0.0,
                1.0 if aircraft["phase"] == "departure" else 0.0,
                1.0 if aircraft["phase"] == "taxi" else 0.0,
            ]
            node_features.append(features)

        # Create edges based on proximity
        edges = []
        for i in range(num_aircraft):
            for j in range(i + 1, num_aircraft):
                # Simple distance calculation
                lat_diff = aircraft_data[i]["lat"] - aircraft_data[j]["lat"]
                lon_diff = aircraft_data[i]["lon"] - aircraft_data[j]["lon"]
                dist = (lat_diff**2 + lon_diff**2) ** 0.5

                if dist < 0.005:  # Close proximity
                    edges.extend([[i, j], [j, i]])

        if len(edges) == 0:
            edges = [[0, 1], [1, 0]]  # Ensure at least one edge

        from torch_geometric.data import Data

        graph_data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edges, dtype=torch.long).t().contiguous(),
        )

        print(
            f"✅ Created graph with {graph_data.x.shape[0]} nodes and {graph_data.edge_index.shape[1]} edges"
        )

        return graph_data, aircraft_data

    except Exception as e:
        print(f"❌ Error creating synthetic data: {e}")
        return None, None


def test_mini_training():
    """Test minimal training loop"""
    print("\n🚀 Testing Mini Training Loop")
    print("-" * 40)

    try:
        # Create model
        model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Generate training data
        training_data = []
        for i in range(20):
            graph_data, aircraft_data = test_synthetic_data()
            if graph_data is not None:
                # Simple bottleneck label based on number of aircraft
                bottleneck_prob = len(aircraft_data) / 10.0  # Normalize
                training_data.append((graph_data, bottleneck_prob))

        print(f"✅ Generated {len(training_data)} training samples")

        # Training loop
        for epoch in range(5):
            epoch_loss = 0
            for graph_data, target in training_data:
                optimizer.zero_grad()

                # Simple prediction: average of node features
                prediction = model(graph_data.x.mean(dim=0, keepdim=True))
                target_tensor = torch.tensor([[target]], dtype=torch.float32)

                loss = criterion(prediction, target_tensor)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(training_data)
            print(f"Epoch {epoch+1}/5 - Loss: {avg_loss:.4f}")

        print("✅ Mini training completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Training error: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    print("🧪 Quick GNN Training Test")
    print("=" * 50)

    # Test 1: Basic GNN functionality
    if not test_basic_gnn():
        return False

    # Test 2: Synthetic data generation
    graph_data, aircraft_data = test_synthetic_data()
    if graph_data is None:
        return False

    # Test 3: Mini training loop
    if not test_mini_training():
        return False

    print("\n🎉 All tests passed! Your GNN training environment is working.")
    print("\n📋 Next Steps:")
    print("1. The basic PyTorch Geometric setup is working")
    print("2. Synthetic data generation is functional")
    print("3. Training loop mechanics are working")
    print("4. You can now debug the full training script")

    return True


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Some tests failed. Check the errors above.")
        sys.exit(1)

