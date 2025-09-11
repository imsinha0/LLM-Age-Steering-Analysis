# Age-Aware Language Model Analysis

A research project investigating age-related patterns and behaviors in large language models, specifically focusing on Llama-3.2-3B. This project explores how language models encode and can be steered based on age-related information through probing, causal interventions, and meta-knowledge analysis.

## 🎯 Project Overview

This repository contains experiments and analyses that examine:
- **Age Probes**: Training linear classifiers to detect age categories from model activations
- **Causal Interventions**: Using learned age directions to steer model behavior
- **Meta-knowledge Testing**: Evaluating model's ability to identify user age from conversations
- **Behavioral Analysis**: Measuring model responses across different age groups

## 📁 Repository Structure

```
experimentAge/
├── src/                          # Source code
│   ├── probes/                   # Age probing experiments
│   │   ├── ageLinearProbeDating.py
│   │   └── predictAgeUsingProbes.py
│   ├── interventions/            # Causal intervention experiments
│   │   └── causalIntervention.py
│   ├── analysis/                 # Analysis and evaluation scripts
│   │   ├── modelMetaKnowledgeChecker.py
│   │   ├── modelMetaKnowledgeDirect.py
│   │   ├── steeredModelMetaKnowledgeChecker.py
│   │   └── steeredModelMetaKnowledgeDirect.py
│   ├── utils/                    # Utility functions
│   │   ├── activationsPCA.py
│   │   ├── cosineSimilarity.py
│   │   ├── measureObstinanceProbabilities.py
│   │   ├── measureObstinancy.py
│   │   └── wordsForEachAge.py
│   └── data_processing/          # Data preparation scripts
│       └── createPersonaSwitchDataset.py
├── data/                         # Data storage
│   ├── raw/                      # Original datasets
│   │   ├── llama_age_1/          # Conversation datasets
│   │   └── causalityQuestions/   # Question prompts
│   └── processed/                # Processed data
│       ├── probe_data*           # Probe training data
│       └── personaSwitchDS*      # Persona switching datasets
├── results/                      # Experiment results
│   ├── probes/                   # Probe training results
│   │   ├── trained_probes*/      # Trained probe models
│   │   └── *.pkl                 # Probe accuracy results
│   └── analysis/                 # Analysis results
│       └── *.csv                 # Accuracy and evaluation results
├── experiments/                  # Experimental notebooks and scripts
│   ├── notebooks/                # Jupyter notebooks
│   │   ├── main.ipynb
│   │   ├── main.py
│   │   └── introToTransformerLens.py
│   └── scripts/                  # Experimental scripts
│       └── graphMaker/           # Visualization scripts
├── docs/                         # Documentation
├── pyproject.toml               # Project configuration
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.13+
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

#### 3. Meta-knowledge Analysis

Test the model's ability to identify user age:

```bash
python src/analysis/modelMetaKnowledgeChecker.py
```

This will:
- Load conversation data from `data/raw/llama_age_1/`
- Test model's age identification capabilities
- Generate accuracy reports by age group

## 🔬 Key Experiments

### Age Probing
- **Purpose**: Understand how age information is encoded in model activations
- **Method**: Train linear classifiers on hidden states from different layers
- **Age Categories**: child, adolescent, adult, older adult
- **Files**: `src/probes/ageLinearProbeDating.py`

### Causal Interventions
- **Purpose**: Test if we can steer model behavior using learned age directions
- **Method**: Add probe directions to residual streams during generation
- **Intervention Strengths**: 0.0 to 10.0
- **Files**: `src/interventions/causalIntervention.py`

### Meta-knowledge Testing
- **Purpose**: Evaluate if models can identify user age from conversations
- **Method**: Prompt models to include secret words based on perceived user age
- **Evaluation**: Compare predicted vs. true age categories
- **Files**: `src/analysis/modelMetaKnowledgeChecker.py`

## 📊 Results

Key findings are stored in the `results/` directory:

- **Probe Accuracy**: `results/probes/probe_accuracy_results3b.pkl`
- **Age-specific Accuracy**: `results/analysis/model_meta_knowledge_*_accuracy_by_age.csv`
- **Intervention Responses**: `results/analysis/OUTDOORACTIVITIEScausal_intervention_responses.csv`

## 🛠️ Development

### Adding New Experiments

1. Create new scripts in appropriate `src/` subdirectories
2. Use the established data loading patterns
3. Save results to `results/` with descriptive names
4. Update this README with new functionality

### Data Format

- **Probe Data**: Pickled dictionaries with 'activations' and 'labels' keys
- **Conversation Data**: Text files with age category in filename
- **Results**: CSV files for tabular data, PKL files for complex objects

## 📝 Dependencies

Key dependencies include:
- `transformer-lens`: For model analysis and hooking
- `torch`: PyTorch for deep learning
- `scikit-learn`: For probe training
- `pandas`: For data manipulation
- `numpy`: For numerical operations

See `requirements.txt` for complete list.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

[Add your license information here]

## 🙏 Acknowledgments

- [TransformerLens](https://github.com/neelnanda-io/TransformerLens) for model analysis tools
- [Meta's Llama models](https://huggingface.co/meta-llama) for the base language model
- Research community for probing and intervention techniques

## 📚 References

[Add relevant papers and references here]

---

For questions or issues, please open an issue on GitHub or contact the maintainers.
