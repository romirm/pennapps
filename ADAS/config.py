"""
ADAS Configuration File
Centralized configuration for the Automated Design of Agentic Systems
"""

import os
from typing import Dict, Any

# Load .env file if available
try:
    from dotenv import load_dotenv

    load_dotenv()  # Load from current directory
except ImportError:
    # dotenv not available, try manual loading
    env_file = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

# API Configuration
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
if not CEREBRAS_API_KEY:
    raise ValueError("CEREBRAS_API_KEY not found in environment variables or .env file")

# Paths
BASE_PATH = "/Users/andyphu/pennapps/pennapps"
ADAS_PATH = os.path.join(BASE_PATH, "ADAS")
VALIDATION_DATASET_PATH = ADAS_PATH
RAW_VALIDATION_PATH = os.path.join(ADAS_PATH, "raw-validation-dataset")
CLEAN_VALIDATION_PATH = os.path.join(ADAS_PATH, "validation-dataset")

# Evolution Parameters
EVOLUTION_CONFIG = {
    "num_cycles": 5,
    "evaluations_per_variant": 3,  # Reduced to respect rate limits
    "max_variants": 5,  # Reduced to manage API calls
    "mock_mode": False,  # Set to True to test without API calls
    "mutation_strategies": [
        "parameter_tuning",
        "prompt_engineering",
        "model_selection",
        "hybrid",
        "novel",
    ],
}

# Model Configuration
MODEL_CONFIG = {
    "default_model": "llama3.1-8b",  # For CleaningAgent
    "task_agent_model": "qwen-3-235b-a22b-thinking-2507",  # Specific model for TaskAgent
    "evaluator_agent_model": "llama-4-scout-17b-16e-instruct",  # Specific model for EvaluatorAgent
    "meta_agent_model": "qwen-3-coder-480b",
    "available_models": [
        "gpt-oss-120b",
        "llama-4-scout-17b-16e-instruct",
        "qwen-3-coder-480b",
        "llama3.1-8b",
        "llama-4-maverick-17b-128e-instruct",
        "llama-3.3-70b",
        "qwen-3-32b",
    ],
    "default_temperature": 0.3,
    "default_max_tokens": 1000,
    "evaluation_temperature": 0.2,
    "mutation_temperature": 0.7,
    "rate_limits": {
        "qwen-3-coder-480b": 10,  # 10 requests per minute
        "qwen-3-235b-a22b-thinking-2507": 30,  # TaskAgent model rate limit
        "gpt-oss-120b": 60,  # Large model, moderate limit
        "llama-4-scout-17b-16e-instruct": 60,  # Similar to other 17b models
        "llama3.1-8b": 120,  # Higher limit for smaller model
        "llama-4-maverick-17b-128e-instruct": 60,
        "default": 30,  # Default rate limit
    },
}

# Task Agent Parameters
TASK_AGENT_CONFIG = {
    "baseline_parameters": {
        "model": "qwen-3-235b-a22b-thinking-2507",  # Use the configured task_agent_model
        "temperature": 0.3,
        "max_tokens": 1000,
        "gnn_hidden_dim": 64,
        "gnn_layers": 4,
        "analysis_approach": "comprehensive",
        "prompt_style": "structured",
    },
    "mutation_ranges": {
        "temperature": (0.1, 0.8),
        "max_tokens": (500, 2000),
        "gnn_hidden_dim": (32, 128),
        "gnn_layers": (2, 6),
    },
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "scoring_criteria": [
        "correctness",
        "safety",
        "efficiency",
        "specificity",
        "timing",
    ],
    "safety_weights": {"safe": 1.0, "questionable": 0.5, "unsafe": 0.0},
    "min_confidence_threshold": 0.3,
}

# Data Cleaning Configuration
CLEANING_CONFIG = {
    "quality_thresholds": {
        "min_confidence": 0.5,
        "min_command_clarity": 0.6,
        "min_completeness": 0.7,
    },
    "rejection_patterns": [
        "unclear transcription",
        "missing callsign",
        "incomplete command",
        "non-operational",
        "weather chatter",
        "administrative",
    ],
}

# Airport-Specific Configuration
AIRPORT_CONFIG = {
    "primary_airport": "KJFK",
    "airport_coordinates": {"KJFK": {"lat": 40.6413, "lon": -73.7781}},
    "runways": {
        "KJFK": ["04L/22R", "04R/22L", "08L/26R", "08R/26L", "13L/31R", "13R/31L"]
    },
    "analysis_radius_nm": 5,
}

# Logging Configuration
LOGGING_CONFIG = {
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": os.path.join(ADAS_PATH, "adas.log"),
    "max_log_size_mb": 50,
    "backup_count": 3,
}

# Performance Monitoring
MONITORING_CONFIG = {
    "save_state_frequency": 5,  # Every N cycles
    "performance_history_limit": 1000,
    "variant_history_limit": 100,
    "evaluation_timeout_seconds": 30,
}


def get_config() -> Dict[str, Any]:
    """Get complete configuration dictionary"""
    return {
        "cerebras_api_key": CEREBRAS_API_KEY,
        "paths": {
            "base": BASE_PATH,
            "adas": ADAS_PATH,
            "validation_dataset": VALIDATION_DATASET_PATH,
            "raw_validation": RAW_VALIDATION_PATH,
            "clean_validation": CLEAN_VALIDATION_PATH,
        },
        "evolution": EVOLUTION_CONFIG,
        "models": MODEL_CONFIG,
        "task_agent": TASK_AGENT_CONFIG,
        "evaluation": EVALUATION_CONFIG,
        "cleaning": CLEANING_CONFIG,
        "airport": AIRPORT_CONFIG,
        "logging": LOGGING_CONFIG,
        "monitoring": MONITORING_CONFIG,
    }


def validate_config() -> bool:
    """Validate configuration settings"""

    # Check required paths exist
    required_paths = [BASE_PATH, ADAS_PATH]
    for path in required_paths:
        if not os.path.exists(path):
            print(f"❌ Required path does not exist: {path}")
            return False

    # Check API key
    if not CEREBRAS_API_KEY or CEREBRAS_API_KEY == "your-api-key-here":
        print("❌ Cerebras API key not configured")
        return False

    # Create validation directories if they don't exist
    os.makedirs(RAW_VALIDATION_PATH, exist_ok=True)
    os.makedirs(CLEAN_VALIDATION_PATH, exist_ok=True)

    print("✅ Configuration validated successfully")
    return True


if __name__ == "__main__":
    # Test configuration
    config = get_config()
    print("ADAS Configuration:")
    print("=" * 40)

    for section, values in config.items():
        print(f"\n{section.upper()}:")
        if isinstance(values, dict):
            for key, value in values.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {values}")

    print("\nValidating configuration...")
    validate_config()
