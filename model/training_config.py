"""
Training Configuration for GNN-KAN Airport Bottleneck Prediction System
"""

# Training hyperparameters
TRAINING_CONFIG = {
    # Model architecture
    "gnn_input_dim": 5,
    "gnn_hidden_dim": 64,
    "gnn_layers": 4,
    "gnn_output_dim": 32,
    "kan_hidden_dim": 64,
    "flight_metadata_dim": 10,
    # Training parameters
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "batch_size": 8,
    "num_epochs": 50,
    "gradient_clip_norm": 1.0,
    # Data generation
    "train_samples": 800,
    "val_samples": 200,
    "test_samples": 100,
    "airports": ["KJFK", "KLAX", "KPHL", "KMIA", "KORD", "KDFW"],
    # Loss function weights
    "loss_weights": {
        "queue": 1.0,
        "intersection": 1.0,
        "gate": 1.0,
        "severity": 0.5,
        "bottleneck_probability": 2.0,
        "resolution_time": 1.5,
        "passengers_affected": 1.0,
        "fuel_waste": 1.0,
        "economic_impact": 0.8,
        "environmental_impact": 0.8,
    },
    # Learning rate scheduling
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "scheduler_min_lr": 1e-6,
    # Early stopping
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 1e-4,
    # Validation and checkpointing
    "validate_every": 1,  # epochs
    "save_every": 5,  # epochs
    "log_every": 10,  # batches
    # Data augmentation
    "noise_std": 0.01,
    "position_jitter": 0.001,  # degrees
    "speed_noise": 0.05,  # fraction
    "altitude_noise": 50,  # feet
    # Synthetic data parameters
    "bottleneck_scenarios": {
        "normal_operations": 0.4,
        "light_congestion": 0.3,
        "moderate_congestion": 0.2,
        "severe_congestion": 0.1,
    },
    "weather_conditions": {
        "clear": 0.6,
        "light_weather": 0.25,
        "moderate_weather": 0.1,
        "severe_weather": 0.05,
    },
    # Aircraft type distribution
    "aircraft_distribution": {
        "B737": 0.25,
        "A320": 0.25,
        "B777": 0.15,
        "A380": 0.05,
        "CRJ9": 0.1,
        "E175": 0.1,
        "B767F": 0.05,
        "B787": 0.05,
    },
    # Phase distribution (varies by scenario)
    "phase_distribution": {
        "normal": {
            "approach": 0.25,
            "departure": 0.25,
            "taxi": 0.3,
            "gate": 0.15,
            "holding": 0.05,
        },
        "congested": {
            "approach": 0.35,
            "departure": 0.2,
            "taxi": 0.25,
            "gate": 0.1,
            "holding": 0.1,
        },
    },
}

# Evaluation metrics configuration
EVALUATION_CONFIG = {
    "metrics": [
        "bottleneck_accuracy",
        "severity_classification_accuracy",
        "resolution_time_mae",
        "passenger_impact_mape",
        "fuel_waste_mape",
        "economic_impact_correlation",
    ],
    "thresholds": {
        "bottleneck_probability": 0.6,
        "high_severity": 0.8,
        "critical_severity": 0.9,
    },
    "tolerance": {
        "resolution_time_minutes": 5,
        "passenger_count": 50,
        "fuel_gallons": 10,
    },
}

# Production deployment configuration
DEPLOYMENT_CONFIG = {
    "model_serving": {
        "batch_size": 1,
        "max_latency_ms": 500,
        "memory_limit_gb": 4,
        "cpu_cores": 2,
    },
    "monitoring": {
        "prediction_drift_threshold": 0.1,
        "performance_degradation_threshold": 0.05,
        "alert_frequency_minutes": 15,
    },
    "data_pipeline": {
        "update_frequency_seconds": 30,
        "data_retention_hours": 168,  # 1 week
        "batch_processing_size": 100,
    },
}

# Quick start configurations for different use cases
QUICK_START_CONFIGS = {
    "development": {
        **TRAINING_CONFIG,
        "train_samples": 100,
        "val_samples": 25,
        "num_epochs": 10,
        "batch_size": 4,
        "log_every": 5,
    },
    "testing": {
        **TRAINING_CONFIG,
        "train_samples": 50,
        "val_samples": 10,
        "num_epochs": 5,
        "batch_size": 2,
        "log_every": 2,
    },
    "production": {
        **TRAINING_CONFIG,
        "train_samples": 2000,
        "val_samples": 500,
        "num_epochs": 100,
        "batch_size": 16,
        "early_stopping_patience": 15,
    },
}

