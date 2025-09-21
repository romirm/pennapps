#!/usr/bin/env python3
"""
Working GNN Training Script for Airport Bottleneck Prediction

This version fixes import issues and provides a working training loop.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import MessagePassing, global_mean_pool
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from dataclasses import dataclass


@dataclass
class AircraftData:
    """Individual aircraft data structure"""

    flight_id: str
    aircraft_type: str
    latitude: float
    longitude: float
    altitude: float
    heading: float
    speed: float
    phase: str  # approach, departure, taxi, gate, holding
    timestamp: str


class BottleneckMessagePassing(MessagePassing):
    """Custom message passing layer for bottleneck detection"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.lin = nn.Linear(in_channels, out_channels)
        self.attention = nn.Linear(out_channels * 2, 1)

    def forward(self, x, edge_index, edge_attr=None):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        # Compute attention weights
        x_concat = torch.cat([x_i, x_j], dim=-1)
        attention_weights = torch.sigmoid(self.attention(x_concat))

        # Apply attention to messages
        if edge_attr is not None:
            messages = self.lin(x_j) * edge_attr.unsqueeze(-1) * attention_weights
        else:
            messages = self.lin(x_j) * attention_weights

        return messages

    def update(self, aggr_out, x):
        return aggr_out + self.lin(x)


class SimpleBottleneckGNN(nn.Module):
    """Simplified GNN for bottleneck detection"""

    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Message passing layers
        self.mp_layers = nn.ModuleList()
        for i in range(num_layers):
            self.mp_layers.append(BottleneckMessagePassing(hidden_dim, hidden_dim))

        # Layer normalization
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

        # Prediction heads
        self.bottleneck_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        # Input projection
        x = self.input_proj(x)

        # Message passing
        for i, layer in enumerate(self.mp_layers):
            residual = x
            x = layer(x, edge_index, edge_attr)
            x = self.layer_norms[i](x)
            x = torch.relu(x)
            x = x + residual  # Residual connection

        # Global pooling for graph-level prediction
        if hasattr(data, "batch"):
            graph_embedding = global_mean_pool(x, data.batch)
        else:
            graph_embedding = torch.mean(x, dim=0, keepdim=True)

        # Bottleneck prediction
        bottleneck_prob = self.bottleneck_predictor(graph_embedding)

        return {
            "bottleneck_probability": bottleneck_prob,
            "node_embeddings": x,
            "graph_embedding": graph_embedding,
        }


class AirportDataset(Dataset):
    """Simplified dataset for airport bottleneck training"""

    def __init__(self, num_samples: int = 100):
        self.num_samples = num_samples
        self.samples = self._generate_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _generate_samples(self):
        """Generate synthetic airport scenarios"""
        samples = []

        print(f"Generating {self.num_samples} synthetic samples...")

        for i in tqdm(range(self.num_samples)):
            # Random scenario parameters
            num_aircraft = random.randint(3, 12)
            bottleneck_severity = random.uniform(0.0, 1.0)

            # Generate aircraft
            aircraft_list = []
            airport_lat, airport_lon = 40.6413, -73.7781  # JFK

            for j in range(num_aircraft):
                aircraft = AircraftData(
                    flight_id=f"SIM{j:03d}",
                    aircraft_type=random.choice(["B737", "A320", "B777"]),
                    latitude=airport_lat + random.uniform(-0.01, 0.01),
                    longitude=airport_lon + random.uniform(-0.01, 0.01),
                    altitude=random.uniform(0, 3000),
                    heading=random.uniform(0, 360),
                    speed=random.uniform(0, 200),
                    phase=random.choice(["approach", "departure", "taxi", "gate"]),
                    timestamp=datetime.now().isoformat(),
                )
                aircraft_list.append(aircraft)

            # Create graph
            graph_data = self._create_graph(aircraft_list)

            # Create label
            label = torch.tensor([bottleneck_severity], dtype=torch.float32)

            samples.append(
                {"graph": graph_data, "label": label, "num_aircraft": num_aircraft}
            )

        return samples

    def _create_graph(self, aircraft_list: List[AircraftData]) -> Data:
        """Create graph from aircraft data"""
        if len(aircraft_list) == 0:
            return Data(
                x=torch.zeros((1, 5)), edge_index=torch.zeros((2, 0), dtype=torch.long)
            )

        # Node features: [altitude, speed, phase_encoded, type_encoded, heading]
        node_features = []
        for aircraft in aircraft_list:
            features = [
                aircraft.altitude / 1000.0,  # Normalize altitude
                aircraft.speed / 100.0,  # Normalize speed
                self._encode_phase(aircraft.phase),
                self._encode_aircraft_type(aircraft.aircraft_type),
                aircraft.heading / 360.0,  # Normalize heading
            ]
            node_features.append(features)

        # Create edges based on proximity
        edges = []
        for i in range(len(aircraft_list)):
            for j in range(i + 1, len(aircraft_list)):
                # Simple distance calculation
                lat_diff = aircraft_list[i].latitude - aircraft_list[j].latitude
                lon_diff = aircraft_list[i].longitude - aircraft_list[j].longitude
                dist = (lat_diff**2 + lon_diff**2) ** 0.5

                if dist < 0.008:  # Within interaction distance
                    edges.extend([[i, j], [j, i]])

        # Ensure at least some connectivity
        if len(edges) == 0 and len(aircraft_list) > 1:
            edges = [[0, 1], [1, 0]]

        edge_index = (
            torch.tensor(edges, dtype=torch.long).t().contiguous()
            if edges
            else torch.zeros((2, 0), dtype=torch.long)
        )

        return Data(
            x=torch.tensor(node_features, dtype=torch.float32), edge_index=edge_index
        )

    def _encode_phase(self, phase: str) -> float:
        """Encode aircraft phase"""
        mapping = {"approach": 0.1, "departure": 0.2, "taxi": 0.3, "gate": 0.4}
        return mapping.get(phase, 0.0)

    def _encode_aircraft_type(self, aircraft_type: str) -> float:
        """Encode aircraft type"""
        mapping = {"B737": 0.1, "A320": 0.2, "B777": 0.3}
        return mapping.get(aircraft_type, 0.1)


def collate_fn(batch):
    """Custom collate function for graph batching"""
    graphs = [item["graph"] for item in batch]
    labels = [item["label"] for item in batch]

    # Batch graphs
    batched_graph = Batch.from_data_list(graphs)
    batched_labels = torch.cat(labels)

    return {"graph": batched_graph, "label": batched_labels}


def train_gnn():
    """Main training function"""
    print("🚀 Starting GNN Training for Airport Bottleneck Prediction")
    print("=" * 60)

    # Configuration
    config = {
        "num_epochs": 20,
        "batch_size": 8,
        "learning_rate": 0.001,
        "train_samples": 200,
        "val_samples": 50,
    }

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    print("📊 Creating datasets...")
    train_dataset = AirportDataset(num_samples=config["train_samples"])
    val_dataset = AirportDataset(num_samples=config["val_samples"])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Create model
    print("🔧 Initializing model...")
    model = SimpleBottleneckGNN(input_dim=5, hidden_dim=64, num_layers=3)
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training history
    train_losses = []
    val_losses = []

    print("🎯 Starting training...")

    for epoch in range(config["num_epochs"]):
        # Training phase
        model.train()
        epoch_train_loss = 0
        num_train_batches = 0

        for batch in tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} - Training"
        ):
            graph_data = batch["graph"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(graph_data)
            predictions = outputs["bottleneck_probability"].squeeze()

            # Handle single sample case
            if predictions.dim() == 0:
                predictions = predictions.unsqueeze(0)
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)

            loss = criterion(predictions, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            num_train_batches += 1

        avg_train_loss = epoch_train_loss / num_train_batches
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                graph_data = batch["graph"].to(device)
                labels = batch["label"].to(device)

                outputs = model(graph_data)
                predictions = outputs["bottleneck_probability"].squeeze()

                # Handle single sample case
                if predictions.dim() == 0:
                    predictions = predictions.unsqueeze(0)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)

                loss = criterion(predictions, labels)
                epoch_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = epoch_val_loss / num_val_batches if num_val_batches > 0 else 0
        val_losses.append(avg_val_loss)

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Print progress
        print(
            f"Epoch {epoch+1}/{config['num_epochs']} - "
            f"Train Loss: {avg_train_loss:.4f}, "
            f"Val Loss: {avg_val_loss:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

    print("✅ Training completed!")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"gnn_bottleneck_model_{timestamp}.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        model_path,
    )
    print(f"💾 Model saved: {model_path}")

    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GNN Training History")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"gnn_training_history_{timestamp}.png")
    print(f"📊 Training plot saved: gnn_training_history_{timestamp}.png")

    # Test model
    print("\n🧪 Testing trained model...")
    model.eval()
    test_dataset = AirportDataset(num_samples=10)
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collate_fn)

    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            graph_data = batch["graph"].to(device)
            true_label = batch["label"].to(device).item()

            outputs = model(graph_data)
            predicted_prob = outputs["bottleneck_probability"].item()

            # Simple accuracy check
            predicted_class = 1 if predicted_prob > 0.5 else 0
            true_class = 1 if true_label > 0.5 else 0

            if predicted_class == true_class:
                correct += 1
            total += 1

            if i < 5:  # Show first 5 examples
                print(
                    f"Sample {i+1}: Predicted={predicted_prob:.3f}, Actual={true_label:.3f}"
                )

    accuracy = correct / total if total > 0 else 0
    print(f"\n📊 Test Accuracy: {accuracy:.2%} ({correct}/{total})")

    return model, train_losses, val_losses


def main():
    """Main function"""
    try:
        model, train_losses, val_losses = train_gnn()

        print("\n🎉 GNN Training Completed Successfully!")
        print(f"📈 Final training loss: {train_losses[-1]:.4f}")
        print(f"📈 Final validation loss: {val_losses[-1]:.4f}")

        print("\n📋 Next Steps:")
        print("1. Your GNN model is trained and saved")
        print("2. You can now integrate it with real ADS-B data")
        print("3. Consider fine-tuning hyperparameters for better performance")
        print("4. Add more sophisticated loss functions for multi-task learning")

    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()

