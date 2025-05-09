# Multi-Modal Context Boundaries (MMCB)

A research project exploring the effectiveness of context boundary implementations against multi-modal prompt injection attacks in large language models.

## Overview

This repository contains a systematic approach to evaluate and compare different context boundary mechanisms in protecting large language models (LLMs) against multi-modal prompt injection attacks. The research addresses four key questions:

1. **Cross-Modal Transfer:** Do context boundaries established in text transfer effectively to other modalities?
2. **Security vs. Complexity:** Which boundary approach provides better cross-modal security with minimal implementation complexity?
3. **Attack Patterns:** Are there systematic patterns in which types of multi-modal attacks succeed against different boundary types?
4. **Model Comparison:** How do the vulnerabilities of different open-source models compare when facing the same multi-modal attacks?

## Key Features

- **Multiple Boundary Mechanisms**: Implementation of token-based, semantic-based, and hybrid boundary approaches
- **Multi-Modal Attack Vectors**: Various attack types combining text with images, structured data, and code
- **Model Comparison**: Evaluation across multiple open-source LLMs (Llama 3 8B, Mistral 7B)
- **Comprehensive Analysis**: Metrics for attack success rates, cross-modal transfer effectiveness, and implementation complexity

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/mmcb-defense.git
cd mmcb-defense

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

```
mmcb-defense/
├── config/               # Configuration files
├── src/                  # Source code
│   ├── boundaries/       # Boundary implementation
│   ├── attacks/          # Attack vector implementations
│   ├── models/           # Model integrations
│   └── utils/            # Utility functions
├── data/                 # Data files and results
├── notebooks/            # Analysis notebooks
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Usage

### Running Experiments

```bash
# Run the default experiment configuration
python src/main.py

# Run with a custom configuration
python src/main.py --config config/custom_experiment.yaml

# Enable debug logging
python src/main.py --log debug
```

### Analyzing Results

The experiment results are saved in the `data/results/` directory. You can analyze them using the provided Jupyter notebooks:

```bash
# Start Jupyter notebook
jupyter notebook notebooks/analysis.ipynb
```

## Boundary Mechanisms

This research evaluates three types of context boundary mechanisms:

1. **Token Boundary**: Uses special tokens to delimit different parts of the prompt (system instructions, user input, etc.)
2. **Semantic Boundary**: Establishes priority levels and explicit role clarifications in natural language
3. **Hybrid Boundary**: Combines both token and semantic approaches for enhanced protection

## Attack Vectors

The project includes various multi-modal attack vectors:

- **Text + Image**: Typographic injections, steganographic content, visual latent alignment
- **Text + Structured Data**: Hidden fields in JSON/CSV, nested structure manipulations
- **Text + Code**: Comment injections, string literal attacks, docstring manipulations

## Research Questions

### 1. Cross-Modal Transfer

This research examines how boundaries established in text transfer to other modalities by measuring:
- Success rates of attacks across different modalities
- Differential performance between text-only and multi-modal contexts
- Boundary enforcement consistency across modalities

### 2. Security vs. Complexity

The project evaluates the trade-off between security and implementation complexity by:
- Comparing attack success rates across boundary types
- Measuring implementation complexity metrics (prompt length, structure complexity)
- Analyzing the security-complexity frontier

### 3. Attack Patterns

The research identifies patterns in successful attacks by:
- Categorizing attack vectors by type, technique, and target
- Analyzing which boundary types are vulnerable to specific attack patterns
- Identifying common characteristics of successful attacks

### 4. Model Comparison

The project compares different open-source models by:
- Evaluating vulnerability profiles across models
- Identifying model-specific weaknesses
- Analyzing the relationship between model size/architecture and vulnerability

## Contributing

Contributions to this research project are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite our work:

```
@misc{mmcb-defense,
  author = {Your Name},
  title = {Multi-Modal Context Boundaries: Defending Against Prompt Injection Attacks},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/mmcb-defense}}
}
```