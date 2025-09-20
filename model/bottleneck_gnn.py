"""
GNN Bottleneck Analyzer

Graph Neural Network model that analyzes spatial relationships to identify 
congestion patterns and critical infrastructure choke points.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data
from typing import Dict, List, Tuple


class BottleneckMessagePassing(MessagePassing):
    """
    Custom message passing layer for bottleneck detection
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr='add')
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


class BottleneckGNN(nn.Module):
    """
    Graph Neural Network for bottleneck detection and analysis
    
    Focuses on:
    - Queue formation detection (runway approaches)
    - Intersection conflict analysis (taxiway crossings)  
    - Gate area congestion modeling
    - Ground traffic flow bottlenecks
    """
    
    def __init__(self, input_dim: int = 5, hidden_dim: int = 64, 
                 num_layers: int = 4, output_dim: int = 32):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Bottleneck detection layers
        self.bottleneck_layers = nn.ModuleList()
        for i in range(num_layers):
            self.bottleneck_layers.append(
                BottleneckMessagePassing(hidden_dim, hidden_dim)
            )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        # Bottleneck classification heads
        self.queue_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.intersection_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.gate_congestion_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Output projection for bottleneck embeddings
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Bottleneck severity predictor
        self.severity_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5),  # 1-5 severity levels
            nn.Softmax(dim=-1)
        )
        
    def forward(self, graph_data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the bottleneck detection GNN
        
        Args:
            graph_data: PyTorch Geometric Data object
            
        Returns:
            Dictionary containing bottleneck predictions and embeddings
        """
        x, edge_index, edge_attr = graph_data.x, graph_data.edge_index, graph_data.edge_attr
        
        # Input projection
        x = self.input_proj(x)
        
        # Message passing layers
        for i, layer in enumerate(self.bottleneck_layers):
            residual = x
            x = layer(x, edge_index, edge_attr)
            x = self.layer_norms[i](x)
            x = F.relu(x)
            x = x + residual  # Residual connection
        
        # Global pooling for graph-level features
        if hasattr(graph_data, 'batch'):
            graph_embedding = global_mean_pool(x, graph_data.batch)
        else:
            graph_embedding = torch.mean(x, dim=0, keepdim=True)
        
        # Bottleneck detection
        bottleneck_predictions = {
            'queue_probability': self.queue_detector(x),
            'intersection_conflict_probability': self.intersection_detector(x),
            'gate_congestion_probability': self.gate_congestion_detector(x),
            'node_embeddings': x,
            'graph_embedding': graph_embedding,
            'bottleneck_embeddings': self.output_proj(x),
            'severity_levels': self.severity_predictor(x)
        }
        
        return bottleneck_predictions
    
    def detect_runway_approach_queues(self, x: torch.Tensor, 
                                    aircraft_phases: List[str]) -> torch.Tensor:
        """
        Detect aircraft queuing on runway approaches
        
        Args:
            x: Node embeddings
            aircraft_phases: List of aircraft operational phases
            
        Returns:
            Queue probability tensor
        """
        approach_mask = torch.tensor([
            1.0 if phase == 'approach' else 0.0 
            for phase in aircraft_phases
        ], device=x.device).unsqueeze(-1)
        
        # Focus on approach aircraft
        approach_embeddings = x * approach_mask
        
        # Detect queue formation patterns
        queue_scores = self.queue_detector(approach_embeddings)
        
        return queue_scores * approach_mask
    
    def detect_taxiway_intersection_conflicts(self, x: torch.Tensor,
                                            aircraft_positions: List[Tuple[float, float]]) -> torch.Tensor:
        """
        Detect conflicts at taxiway intersections
        
        Args:
            x: Node embeddings
            aircraft_positions: List of (lat, lon) positions
            
        Returns:
            Conflict probability tensor
        """
        # Calculate pairwise distances
        conflict_scores = []
        
        for i in range(len(aircraft_positions)):
            for j in range(i + 1, len(aircraft_positions)):
                pos_i = aircraft_positions[i]
                pos_j = aircraft_positions[j]
                
                # Calculate distance (simplified)
                dist = ((pos_i[0] - pos_j[0])**2 + (pos_i[1] - pos_j[1])**2)**0.5
                
                if dist < 0.001:  # Close proximity threshold
                    # Combine embeddings for conflict detection
                    combined_embedding = torch.cat([x[i], x[j]], dim=-1)
                    conflict_score = self.intersection_detector(combined_embedding)
                    conflict_scores.append(conflict_score)
        
        if len(conflict_scores) == 0:
            return torch.zeros(x.size(0), 1, device=x.device)
        
        return torch.stack(conflict_scores)
    
    def detect_gate_area_congestion(self, x: torch.Tensor,
                                  gate_assignments: List[str]) -> torch.Tensor:
        """
        Detect congestion in gate areas
        
        Args:
            x: Node embeddings
            gate_assignments: List of gate assignments
            
        Returns:
            Congestion probability tensor
        """
        # Group aircraft by gate area
        gate_groups = {}
        for i, gate in enumerate(gate_assignments):
            gate_area = gate.split('-')[0] if '-' in gate else gate
            if gate_area not in gate_groups:
                gate_groups[gate_area] = []
            gate_groups[gate_area].append(i)
        
        congestion_scores = []
        
        for gate_area, aircraft_indices in gate_groups.items():
            if len(aircraft_indices) > 1:  # Multiple aircraft in same gate area
                # Average embeddings for gate area
                gate_embedding = torch.mean(x[aircraft_indices], dim=0, keepdim=True)
                congestion_score = self.gate_congestion_detector(gate_embedding)
                congestion_scores.extend([congestion_score] * len(aircraft_indices))
            else:
                congestion_scores.append(torch.zeros(1, 1, device=x.device))
        
        return torch.cat(congestion_scores, dim=0)
    
    def predict_bottleneck_evolution(self, current_embeddings: torch.Tensor,
                                   time_horizon_minutes: int = 15) -> Dict[str, torch.Tensor]:
        """
        Predict how bottlenecks will evolve over time
        
        Args:
            current_embeddings: Current bottleneck embeddings
            time_horizon_minutes: Prediction time horizon
            
        Returns:
            Dictionary with temporal bottleneck predictions
        """
        # Simple temporal prediction (in production, use LSTM/Transformer)
        temporal_weights = torch.linspace(1.0, 0.5, time_horizon_minutes, device=current_embeddings.device)
        
        predictions = {
            'bottleneck_intensity': current_embeddings.mean(dim=0) * temporal_weights.unsqueeze(-1),
            'resolution_probability': torch.sigmoid(current_embeddings.mean(dim=0) * -1.0),
            'escalation_risk': torch.sigmoid(current_embeddings.mean(dim=0) * 0.5)
        }
        
        return predictions
