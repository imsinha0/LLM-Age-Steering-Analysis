"""
Configuration file for age-aware-lm-analysis project.
Contains default settings and paths used across the project.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SRC_DIR = PROJECT_ROOT / "src"

# Data paths
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROBE_DATA_DIR = PROCESSED_DATA_DIR
CONVERSATION_DATA_DIR = RAW_DATA_DIR / "llama_age_1"

# Results paths
PROBE_RESULTS_DIR = RESULTS_DIR / "probes"
ANALYSIS_RESULTS_DIR = RESULTS_DIR / "analysis"

# Model configuration
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-3B"
DEFAULT_DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"

# Age categories
AGE_CATEGORIES = ["child", "adolescent", "adult", "older adult"]

# Probe configuration
DEFAULT_PROBING_LAYER = 20
NUM_LAYERS = 28

# Training configuration
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_MAX_ITER = 1000

# Generation configuration
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9

# Intervention configuration
INTERVENTION_STRENGTHS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

# Secret words for meta-knowledge testing
SECRET_WORDS = {
    "child": "zxcfrf",
    "adolescent": "bnrfrm", 
    "adult": "ghdcjk",
    "older_adult": "gdhije",
}

# File patterns
CONVERSATION_FILE_PATTERN = "conversation_*.txt"
PROBE_FILE_PATTERN = "probe_trained_age_{}.pkl"
RESULTS_FILE_PATTERN = "probe_accuracy_results{}.pkl"
