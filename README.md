# Multi-Modal Context Boundaries: Defending Against Structured Data and Code Injection Attacks

A quarter-long research project for [CSE227](https://plsyssec.github.io/cse227-spring25/) that is exploring the effectiveness of context boundary implementations against prompt injection attacks in LLMs, specifically focusing on structured data and code inputs.

## Overview

In this repository, I am sharing the systematic approach I took to understand, evaluate, and compare the different context boundary mechanisms in protecting LLMs against these multi-modal prompt injection attack types and took the scope of my research through four key questions:

1. Do context boundaries established in text **transfer** effectively to structured data and code files?
2. Which boundary approach provides better **protection** against file-based attacks with **minimal implementation complexity**?
3. Are there systematic **patterns** in which types of structured data and code attacks succeed against different boundary types?
4. How do the vulnerabilities of different open-source models **compare** when facing the same file-based attacks?

## Key Features

- **Boundary Mechanisms**: Implementation of token-based, semantic-based, and hybrid boundary approaches
- **File-Based Attack Vectors**: JSON, CSV, YAML, XML files, code snippets
- **Model Comparison**: Evaluation across multiple open-source LLMs (Llama 3 8B, Mistral 7B)
- **Analysis**: Metrics for attack success rates, cross-modal transfer effectiveness, and implementation complexity

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
│   ├── attacks/          # Attack vector implementations (structured data & code)
│   ├── models/           # Model integrations
│   └── utils/            # Utility functions
├── data/                 # Data files and results
├── notebooks/            # Analysis notebooks
├── tests/                # Unit tests
└── docs/                 # Documentation
```

## Usage

## Usage

### Running Experiments

```bash
# Run the default experiment configuration
python src/main.py

# Run with debug logging enabled
python src/main.py --log debug

# Run with a custom configuration file
python src/main.py --config config/custom_experiment.yaml

# Run with specific attack types only
python src/main.py --attack_types json csv yaml

# Run in quick test mode (fewer attacks)
python src/main.py --quick

# Run with custom batch size for parallel processing
python src/main.py --batch_size 8

# Resume from a previous checkpoint
python src/main.py --resume data/checkpoints/checkpoint_20250519_153024.json

# Run specific attack types with debug logging
python src/main.py --attack_types json python --log debug

# Run a quick test with a custom configuration
python src/main.py --config config/custom_experiment.yaml --quick

### Analyzing Results

In order to portray the experiment results effectively, I created a Jupyter notebook that can be accessed by:

```bash
# Start Jupyter notebook
jupyter notebook notebooks/analysis.ipynb
```

## Boundary Mechanisms

1. **Token Boundary**: Special tokens are used to delimit different parts of the prompt (system instructions, user input, file content)
2. **Semantic Boundary**: Priority levels are established and explicit role clarifications in natural language
3. **Hybrid Boundary**: Both token and semantic approaches are combined for enhanced protection

## Attack Vectors Used

- **Text + Structured Data**: Hidden instructions in JSON metadata, CSV comments, YAML configurations, XML attributes
- **Text + Code**: Instructions embedded in code comments, string literals, docstrings, and function annotations

## Citation

If you use this code in your research, please cite our work:

```
@misc{mmcb-defense,
  author = {Arnav Khinvasara},
  title = {Multi-Modal Context Boundaries: Defending Against Structured Data and Code Injection Attacks},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/khinvi/mmcb-defense}}
}
```

## Acknowledgments

[Professor Deian Stefan](https://cseweb.ucsd.edu/~dstefan/)
