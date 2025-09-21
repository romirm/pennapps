"""
GNN Training Script for Airport Bottleneck Prediction System

This script implements the training pipeline for the hybrid GNN-KAN model
with synthetic data generation, loss functions, and evaluation metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import json
import random
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from .airport_bottleneck_model import AirportBottleneckModel
from .config import BOTTLENECK_CONFIG
from .adsb_processor import ADSBDataProcessor, AircraftData


class BottleneckDataset(Dataset):
    """
    Dataset for training bottleneck prediction models
    Generates synthetic airport scenarios with known bottleneck labels
    """

    def __init__(self, num_samples: int = 1000, airports: List[str] = None):
        self.num_samples = num_samples
        self.airports = airports or ["KJFK", "KLAX", "KPHL", "KMIA"]
        self.adsb_processor = ADSBDataProcessor(BOTTLENECK_CONFIG)

        # Generate synthetic scenarios
        print(f"Generating {num_samples} synthetic airport scenarios...")
        self.scenarios = self._generate_scenarios()

    def __len__(self):
        return len(self.scenarios)

    def __getitem__(self, idx):
        scenario = self.scenarios[idx]
        return {
            "graph_data": scenario["graph_data"],
            "labels": scenario["labels"],
            "metadata": scenario["metadata"],
        }

    def _generate_scenarios(self) -> List[Dict]:
        """Generate synthetic airport scenarios with varying bottleneck conditions"""
        scenarios = []

        for i in tqdm(range(self.num_samples), desc="Generating scenarios"):
            airport = random.choice(self.airports)

            # Generate random scenario parameters
            num_aircraft = random.randint(5, 25)
            bottleneck_severity = random.uniform(0.0, 1.0)
            weather_impact = random.uniform(0.8, 1.2)  # Weather multiplier

            # Create synthetic aircraft data
            aircraft_list = self._generate_aircraft_scenario(
                airport, num_aircraft, bottleneck_severity, weather_impact
            )

            # Build graph
            airport_config = self._get_airport_config(airport)
            graph_data = self.adsb_processor.construct_bottleneck_graph(
                aircraft_list, airport_config
            )

            # Generate labels
            labels = self._generate_labels(
                aircraft_list, bottleneck_severity, weather_impact
            )

            # Metadata for KAN predictor
            metadata = self._generate_flight_metadata(aircraft_list)

            scenarios.append(
                {
                    "graph_data": graph_data,
                    "labels": labels,
                    "metadata": metadata,
                    "airport": airport,
                    "scenario_params": {
                        "num_aircraft": num_aircraft,
                        "bottleneck_severity": bottleneck_severity,
                        "weather_impact": weather_impact,
                    },
                }
            )

        return scenarios

    def _generate_aircraft_scenario(
        self,
        airport: str,
        num_aircraft: int,
        bottleneck_severity: float,
        weather_impact: float,
    ) -> List[AircraftData]:
        """Generate synthetic aircraft data for a scenario"""
        aircraft_list = []
        airport_coords = self._get_airport_coordinates(airport)

        if not airport_coords:
            return aircraft_list

        airport_lat, airport_lon = airport_coords

        # Aircraft types with probabilities
        aircraft_types = ["B737", "A320", "B777", "A380", "CRJ9", "E175", "B767F"]
        type_weights = [0.3, 0.25, 0.15, 0.05, 0.1, 0.1, 0.05]

        # Phases with probabilities (adjusted by bottleneck severity)
        phases = ["approach", "departure", "taxi", "gate", "holding"]
        if bottleneck_severity > 0.7:
            # More congested scenario
            phase_weights = [0.3, 0.2, 0.25, 0.15, 0.1]
        else:
            # Normal operations
            phase_weights = [0.25, 0.25, 0.3, 0.15, 0.05]

        for i in range(num_aircraft):
            # Random position within airport area (±0.02 degrees ~ 2km)
            lat_offset = random.uniform(-0.02, 0.02)
            lon_offset = random.uniform(-0.02, 0.02)

            aircraft_type = random.choices(aircraft_types, weights=type_weights)[0]
            phase = random.choices(phases, weights=phase_weights)[0]

            # Generate realistic parameters based on phase
            if phase == "approach":
                altitude = random.uniform(500, 3000)
                speed = random.uniform(120, 180)
                heading = random.uniform(0, 360)
            elif phase == "departure":
                altitude = random.uniform(0, 2000)
                speed = random.uniform(80, 150)
                heading = random.uniform(0, 360)
            elif phase == "taxi":
                altitude = random.uniform(0, 50)
                speed = random.uniform(5, 25)
                heading = random.uniform(0, 360)
            elif phase == "gate":
                altitude = 0
                speed = 0
                heading = random.uniform(0, 360)
            else:  # holding
                altitude = random.uniform(1000, 5000)
                speed = random.uniform(200, 250)
                heading = random.uniform(0, 360)

            # Apply weather impact
            speed *= weather_impact

            aircraft = AircraftData(
                flight_id=f"SIM{i:03d}",
                aircraft_type=aircraft_type,
                latitude=airport_lat + lat_offset,
                longitude=airport_lon + lon_offset,
                altitude=altitude,
                heading=heading,
                speed=speed,
                phase=phase,
                timestamp=datetime.now().isoformat(),
            )

            aircraft_list.append(aircraft)

        return aircraft_list

    def _generate_labels(
        self,
        aircraft_list: List[AircraftData],
        bottleneck_severity: float,
        weather_impact: float,
    ) -> Dict[str, torch.Tensor]:
        """Generate ground truth labels for training"""
        num_aircraft = len(aircraft_list)

        # Node-level labels
        queue_probabilities = []
        intersection_conflicts = []
        gate_congestion = []
        severity_levels = []

        for aircraft in aircraft_list:
            # Queue probability based on phase and severity
            if aircraft.phase == "approach":
                queue_prob = min(
                    bottleneck_severity * 0.8 + random.uniform(-0.1, 0.1), 1.0
                )
            elif aircraft.phase == "departure":
                queue_prob = min(
                    bottleneck_severity * 0.6 + random.uniform(-0.1, 0.1), 1.0
                )
            else:
                queue_prob = min(
                    bottleneck_severity * 0.3 + random.uniform(-0.05, 0.05), 1.0
                )

            # Intersection conflicts
            if aircraft.phase == "taxi":
                intersection_prob = min(
                    bottleneck_severity * 0.7 + random.uniform(-0.1, 0.1), 1.0
                )
            else:
                intersection_prob = min(
                    bottleneck_severity * 0.2 + random.uniform(-0.05, 0.05), 1.0
                )

            # Gate congestion
            if aircraft.phase == "gate":
                gate_prob = min(
                    bottleneck_severity * 0.9 + random.uniform(-0.1, 0.1), 1.0
                )
            else:
                gate_prob = min(
                    bottleneck_severity * 0.1 + random.uniform(-0.05, 0.05), 1.0
                )

            # Severity level (1-5)
            severity = (
                int(min(bottleneck_severity * 5 + random.uniform(-0.5, 0.5), 5)) + 1
            )
            severity = max(1, min(5, severity))  # Clamp to 1-5

            queue_probabilities.append(queue_prob)
            intersection_conflicts.append(intersection_prob)
            gate_congestion.append(gate_prob)
            severity_levels.append(severity - 1)  # 0-indexed for cross-entropy

        # Impact prediction labels (for KAN)
        impact_labels = {
            "bottleneck_probability": torch.tensor(
                [bottleneck_severity], dtype=torch.float32
            ),
            "resolution_time_minutes": torch.tensor(
                [bottleneck_severity * 30 + random.uniform(-5, 5)], dtype=torch.float32
            ),
            "passengers_affected": torch.tensor(
                [num_aircraft * 150 * bottleneck_severity + random.uniform(-50, 50)],
                dtype=torch.float32,
            ),
            "fuel_waste_gallons": torch.tensor(
                [num_aircraft * bottleneck_severity * 25 + random.uniform(-10, 10)],
                dtype=torch.float32,
            ),
            "economic_impact": torch.tensor(
                [bottleneck_severity * 5000 + random.uniform(-1000, 1000)],
                dtype=torch.float32,
            ),
            "environmental_impact": torch.tensor(
                [bottleneck_severity * 1000 + random.uniform(-200, 200)],
                dtype=torch.float32,
            ),
        }

        return {
            "queue_probability": torch.tensor(queue_probabilities, dtype=torch.float32),
            "intersection_conflict_probability": torch.tensor(
                intersection_conflicts, dtype=torch.float32
            ),
            "gate_congestion_probability": torch.tensor(
                gate_congestion, dtype=torch.float32
            ),
            "severity_levels": torch.tensor(severity_levels, dtype=torch.long),
            "impact_labels": impact_labels,
        }

    def _generate_flight_metadata(
        self, aircraft_list: List[AircraftData]
    ) -> torch.Tensor:
        """Generate flight metadata tensor for KAN predictor"""
        metadata_list = []

        for aircraft in aircraft_list:
            # Simplified metadata features
            features = [
                self._estimate_passengers(aircraft.aircraft_type) / 1000.0,  # Normalize
                self._estimate_fuel_capacity(aircraft.aircraft_type)
                / 10000.0,  # Normalize
                aircraft.altitude / 1000.0,
                aircraft.speed / 100.0,
                aircraft.heading / 360.0,
                self._encode_phase(aircraft.phase),
                self._encode_aircraft_type(aircraft.aircraft_type),
                random.uniform(0.7, 0.9),  # Load factor
                random.uniform(0.8, 1.2),  # Weather factor
                random.uniform(1000, 8000) / 10000.0,  # Range (normalized)
            ]
            metadata_list.append(features)

        # Handle empty aircraft list
        if len(metadata_list) == 0:
            return torch.zeros((1, 10))

        return torch.tensor(metadata_list, dtype=torch.float32)

    def _estimate_passengers(self, aircraft_type: str) -> int:
        """Estimate passenger count by aircraft type"""
        passenger_capacity = {
            "B737": 150,
            "A320": 180,
            "B777": 350,
            "A380": 550,
            "CRJ9": 76,
            "E175": 88,
            "B767F": 0,
        }
        capacity = passenger_capacity.get(aircraft_type, 150)
        return int(capacity * random.uniform(0.7, 0.9))  # Load factor

    def _estimate_fuel_capacity(self, aircraft_type: str) -> int:
        """Estimate fuel capacity by aircraft type"""
        fuel_capacity = {
            "B737": 6875,
            "A320": 6400,
            "B777": 45220,
            "A380": 84535,
            "CRJ9": 2750,
            "E175": 2850,
            "B767F": 16700,
        }
        return fuel_capacity.get(aircraft_type, 6875)

    def _encode_phase(self, phase: str) -> float:
        """Encode aircraft phase"""
        phase_mapping = {
            "approach": 0.1,
            "departure": 0.2,
            "taxi": 0.3,
            "gate": 0.4,
            "holding": 0.5,
        }
        return phase_mapping.get(phase, 0.0)

    def _encode_aircraft_type(self, aircraft_type: str) -> float:
        """Encode aircraft type"""
        type_mapping = {
            "B737": 0.1,
            "A320": 0.2,
            "B777": 0.3,
            "A380": 0.4,
            "CRJ9": 0.5,
            "E175": 0.6,
            "B767F": 0.7,
        }
        return type_mapping.get(aircraft_type, 0.0)

    def _get_airport_coordinates(
        self, airport_icao: str
    ) -> Optional[Tuple[float, float]]:
        """Get airport coordinates"""
        airport_coords = {
            "KJFK": (40.6413, -73.7781),
            "KLAX": (33.9425, -118.4081),
            "KPHL": (39.8729, -75.2407),
            "KMIA": (25.7959, -80.2870),
        }
        return airport_coords.get(airport_icao)

    def _get_airport_config(self, airport_icao: str) -> Dict:
        """Get airport configuration"""
        return {
            "icao": airport_icao,
            "name": f"{airport_icao} Airport",
            "runways": [
                {"id": "09L/27R", "length": 4000, "width": 45, "heading": 90},
                {"id": "09R/27L", "length": 4000, "width": 45, "heading": 90},
            ],
            "gates": ["A1", "A2", "B1", "B2", "C1", "C2"],
            "taxiways": ["A", "B", "C", "D"],
        }


class BottleneckLoss(nn.Module):
    """
    Combined loss function for bottleneck prediction
    """

    def __init__(self, weights: Dict[str, float] = None):
        super().__init__()
        self.weights = weights or {
            "queue": 1.0,
            "intersection": 1.0,
            "gate": 1.0,
            "severity": 0.5,
            "impact": 2.0,
        }

        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(
        self, predictions: Dict[str, torch.Tensor], labels: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Calculate combined loss"""
        losses = {}

        # GNN bottleneck detection losses
        if "queue_probability" in predictions:
            losses["queue_loss"] = self.bce_loss(
                predictions["queue_probability"].squeeze(), labels["queue_probability"]
            )

        if "intersection_conflict_probability" in predictions:
            losses["intersection_loss"] = self.bce_loss(
                predictions["intersection_conflict_probability"].squeeze(),
                labels["intersection_conflict_probability"],
            )

        if "gate_congestion_probability" in predictions:
            losses["gate_loss"] = self.bce_loss(
                predictions["gate_congestion_probability"].squeeze(),
                labels["gate_congestion_probability"],
            )

        if "severity_levels" in predictions:
            losses["severity_loss"] = self.ce_loss(
                predictions["severity_levels"], labels["severity_levels"]
            )

        # KAN impact prediction losses
        impact_labels = labels.get("impact_labels", {})
        for impact_type in [
            "bottleneck_probability",
            "resolution_time_minutes",
            "passengers_affected",
            "fuel_waste_gallons",
            "economic_impact",
            "environmental_impact",
        ]:
            if impact_type in predictions and impact_type in impact_labels:
                losses[f"{impact_type}_loss"] = self.mse_loss(
                    predictions[impact_type].squeeze(),
                    impact_labels[impact_type].squeeze(),
                )

        # Combined loss
        total_loss = 0
        for loss_name, loss_value in losses.items():
            if "queue" in loss_name:
                total_loss += self.weights["queue"] * loss_value
            elif "intersection" in loss_name:
                total_loss += self.weights["intersection"] * loss_value
            elif "gate" in loss_name:
                total_loss += self.weights["gate"] * loss_value
            elif "severity" in loss_name:
                total_loss += self.weights["severity"] * loss_value
            else:
                total_loss += self.weights["impact"] * loss_value

        losses["total_loss"] = total_loss
        return losses


class BottleneckTrainer:
    """
    Trainer class for the bottleneck prediction model
    """

    def __init__(self, model: AirportBottleneckModel, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer and loss
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 0.001),
            weight_decay=config.get("weight_decay", 0.01),
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=5, factor=0.5
        )

        self.loss_fn = BottleneckLoss(config.get("loss_weights"))

        # Training history
        self.train_history = {"loss": [], "val_loss": []}

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
            # Move data to device
            graph_data = batch["graph_data"].to(self.device)

            # Handle labels - some might be nested dictionaries
            labels = {}
            for k, v in batch["labels"].items():
                if isinstance(v, dict):
                    labels[k] = {
                        sub_k: sub_v.to(self.device) for sub_k, sub_v in v.items()
                    }
                else:
                    labels[k] = v.to(self.device)

            metadata = batch["metadata"].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            # Get GNN predictions
            gnn_predictions = self.model.gnn(graph_data)

            # Get KAN predictions
            kan_predictions = self.model.kan_predictor(
                gnn_predictions["bottleneck_embeddings"], metadata
            )

            # Combine predictions
            all_predictions = {**gnn_predictions, **kan_predictions}

            # Calculate loss
            losses = self.loss_fn(all_predictions, labels)
            total_loss = losses["total_loss"]

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_losses.append(total_loss.item())

            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {total_loss.item():.4f}")

        return {"train_loss": np.mean(epoch_losses)}

    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                # Move data to device
                graph_data = batch["graph_data"].to(self.device)

                # Handle labels - some might be nested dictionaries
                labels = {}
                for k, v in batch["labels"].items():
                    if isinstance(v, dict):
                        labels[k] = {
                            sub_k: sub_v.to(self.device) for sub_k, sub_v in v.items()
                        }
                    else:
                        labels[k] = v.to(self.device)

                metadata = batch["metadata"].to(self.device)

                # Forward pass
                gnn_predictions = self.model.gnn(graph_data)
                kan_predictions = self.model.kan_predictor(
                    gnn_predictions["bottleneck_embeddings"], metadata
                )

                all_predictions = {**gnn_predictions, **kan_predictions}

                # Calculate loss
                losses = self.loss_fn(all_predictions, labels)
                val_losses.append(losses["total_loss"].item())

        return {"val_loss": np.mean(val_losses)}

    def train(
        self,
        train_dataset: BottleneckDataset,
        val_dataset: BottleneckDataset,
        num_epochs: int = 50,
        batch_size: int = 8,
    ):
        """Main training loop"""

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

        print(f"Starting training on {self.device}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        best_val_loss = float("inf")

        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)

            # Train
            train_metrics = self.train_epoch(train_loader)

            # Validate
            val_metrics = self.validate(val_loader)

            # Update history
            self.train_history["loss"].append(train_metrics["train_loss"])
            self.train_history["val_loss"].append(val_metrics["val_loss"])

            # Print metrics
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Learning rate scheduling
            self.scheduler.step(val_metrics["val_loss"])

            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_model("best_bottleneck_model.pth")
                print("✅ New best model saved!")

        print("\n🎉 Training completed!")
        return self.train_history

    def _collate_fn(self, batch):
        """Custom collate function for graph data"""
        # Handle empty batch
        if len(batch) == 0:
            return {
                "graph_data": Data(
                    x=torch.zeros((1, 5)),
                    edge_index=torch.zeros((2, 0), dtype=torch.long),
                ),
                "labels": {"queue_probability": torch.zeros(1)},
                "metadata": torch.zeros((1, 10)),
            }

        graph_data_list = [item["graph_data"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        metadata_list = [item["metadata"] for item in batch]

        # Batch graphs - handle empty graphs
        valid_graphs = [g for g in graph_data_list if g.x.size(0) > 0]
        if len(valid_graphs) == 0:
            # Create dummy graph if all are empty
            batched_graphs = Data(
                x=torch.zeros((1, 5)),
                edge_index=torch.zeros((2, 0), dtype=torch.long),
                batch=torch.zeros(1, dtype=torch.long),
            )
        else:
            batched_graphs = Batch.from_data_list(valid_graphs)

        # Combine labels
        combined_labels = {}
        if len(labels_list) > 0:
            for key in labels_list[0].keys():
                if key != "impact_labels":
                    # Concatenate node-level labels
                    label_tensors = [
                        labels[key] for labels in labels_list if key in labels
                    ]
                    if len(label_tensors) > 0:
                        combined_labels[key] = torch.cat(label_tensors)
                    else:
                        combined_labels[key] = torch.zeros(1)
                else:
                    # Handle impact labels separately (graph-level)
                    impact_labels = {}
                    for impact_key in labels_list[0]["impact_labels"].keys():
                        impact_tensors = [
                            labels["impact_labels"][impact_key]
                            for labels in labels_list
                            if "impact_labels" in labels
                            and impact_key in labels["impact_labels"]
                        ]
                        if len(impact_tensors) > 0:
                            impact_labels[impact_key] = torch.cat(impact_tensors)
                        else:
                            impact_labels[impact_key] = torch.zeros(1)
                    combined_labels["impact_labels"] = impact_labels

        # Combine metadata
        if len(metadata_list) > 0:
            # Handle variable-size metadata by taking mean over nodes for each graph
            processed_metadata = []
            for metadata in metadata_list:
                if metadata.dim() == 2:  # [num_nodes, features]
                    processed_metadata.append(
                        metadata.mean(dim=0)
                    )  # Average over nodes
                else:  # [features] - already averaged
                    processed_metadata.append(metadata)
            combined_metadata = torch.stack(processed_metadata)
        else:
            combined_metadata = torch.zeros((len(batch), 10))

        return {
            "graph_data": batched_graphs,
            "labels": combined_labels,
            "metadata": combined_metadata,
        }

    def save_model(self, filepath: str):
        """Save the trained model"""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
                "train_history": self.train_history,
            },
            filepath,
        )

    def load_model(self, filepath: str):
        """Load a trained model"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.train_history = checkpoint.get(
            "train_history", {"loss": [], "val_loss": []}
        )

    def plot_training_history(self):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_history["loss"], label="Training Loss")
        plt.plot(self.train_history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training History")
        plt.legend()
        plt.grid(True)
        plt.savefig("training_history.png")
        plt.show()


def main():
    """Main training function"""
    print("🚀 GNN-KAN Airport Bottleneck Training System")
    print("=" * 60)

    # Training configuration
    training_config = {
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "batch_size": 8,
        "num_epochs": 50,
        "loss_weights": {
            "queue": 1.0,
            "intersection": 1.0,
            "gate": 1.0,
            "severity": 0.5,
            "impact": 2.0,
        },
    }

    # Create datasets
    print("📊 Creating training datasets...")
    train_dataset = BottleneckDataset(
        num_samples=800, airports=["KJFK", "KLAX", "KPHL", "KMIA"]
    )
    val_dataset = BottleneckDataset(num_samples=200, airports=["KJFK", "KLAX"])

    # Initialize model
    print("🔧 Initializing model...")
    model = AirportBottleneckModel(BOTTLENECK_CONFIG)

    # Create trainer
    trainer = BottleneckTrainer(model, training_config)

    # Train model
    print("🎯 Starting training...")
    history = trainer.train(
        train_dataset,
        val_dataset,
        num_epochs=training_config["num_epochs"],
        batch_size=training_config["batch_size"],
    )

    # Plot results
    trainer.plot_training_history()

    # Save final model
    trainer.save_model("final_bottleneck_model.pth")

    print("✅ Training completed successfully!")
    print("📁 Models saved: best_bottleneck_model.pth, final_bottleneck_model.pth")


if __name__ == "__main__":
    main()
