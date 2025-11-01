# Age-Aware Language Model Analysis

A research project investigating age-related patterns and behaviors in large language models, specifically focusing on Llama-3.2-3B. This project explores how language models encode and can be steered based on age-related information through probing, causal interventions, and meta-knowledge analysis.

## Project Overview

This repository contains experiments and analyses that examine:
- **Age Probes**: Training linear classifiers to detect age categories from model activations
- **Causal Interventions**: Using learned age directions to steer model behavior
- **Meta-knowledge Testing**: Evaluating model's ability to identify user age from conversations
- **Behavioral Analysis**: Measuring model responses across different age groups


## Quick Start

### Prerequisites

- Python 3.12
- CUDA-compatible GPU (recommended)
- Sufficient disk space for model weights and data

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd experimentAge
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up the environment:
```bash
# For development installation
pip install -e .
```

### Basic Usage

#### 1. Training Age Probes

Train linear probes to detect age categories from model activations:

```bash
python src/probes/ageLinearProbeDating.py
```

This will:
- Load activation data from `data/processed/probe_data_3b`
- Train logistic regression probes for each layer
- Save trained probes to `results/probes/trained_probes3b/`

#### 2. Running Causal Interventions

Use trained probes to steer model behavior:

```bash
python src/interventions/causalIntervention.py
```

This will:
- Load a trained probe from layer 20
- Apply interventions with different strengths
- Generate responses for different age categories
- Save results to `results/analysis/`


## Acknowledgments

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for model analysis tools
- [Meta's Llama models](https://huggingface.co/meta-llama) for the base language model
