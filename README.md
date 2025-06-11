# Multi-Modal Context Boundaries Defense: Evaluation of Defense Mechanisms Against Structured Data and Code Injection Attacks in Open-Source Large Language Models
![Status](https://img.shields.io/badge/Status-In_Progress-yellow)

A quarter-long research project for [CSE227](https://plsyssec.github.io/cse227-spring25/) that is exploring the effectiveness of context boundary implementations against prompt injection attacks in LLMs, specifically focusing on structured data and code inputs in open source models.

## Overview

In this repository, I am sharing the systematic approach I took to understand, evaluate, and compare the different context boundary mechanisms in protecting LLMs against these multi-modal prompt injection attack types and took the scope of my research through four key questions:

1. Do textual context boundaries methods work with the structured data and code files too? 
2. Which specific boundary implementations provide protection against file-based attacks but also respects minimal implementation complexity and to what extent?
3. Are there noticeable systematic patterns where certain types of structured data as well as code attacks are more successful than other boundary types? 
4. Even when facing the same types of attacks, how do different open-source models compare and contrast?

## Key Features

- **Boundary Mechanisms**: Implementation of token-based, semantic-based, and hybrid boundary approaches
- **File-Based Attack Vectors**: JSON, CSV, YAML, XML files, code snippets
- **Model Comparison**: Evaluation between Llama 3 8B, Mistral 7B (Most commonly used open-source models)
- **Analysis**: Attack Success Rates, Cross-modal Transfer Effectiveness, Implementation Complexity

## Installation

```bash
# Clone the repository
git clone https://github.com/khinvi/mmcb-defense.git
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

```

### Analyzing Results

Visualize the data with the Jupyter Notebook I created; use these instructions to set it up:

```bash
# Start Jupyter notebook
jupyter notebook notebooks/analysis.ipynb
```

## Boundary Mechanisms

1. **Token Boundary**: Special tokens as delimiters fir different parts of the prompt (system instructions, user input, file content)
2. **Semantic Boundary**: Priority levels + explicit role clarifications in natural language
3. **Hybrid Boundary**: Token + semantic approaches

## Attack Vectors Used

- **Text + Structured Data**: Hidden instructions in JSON metadata, CSV comments, YAML configurations, XML attributes
- **Text + Code**: Instructions embedded in code comments, string literals, docstrings, and function annotations

## Citation

If you use this code in your research, please cite my work:

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

- [Professor Deian Stefan](https://cseweb.ucsd.edu/~dstefan/)
- LLM Usage: Claude 4.0 Opus, Claude 4.0 Sonnet, Claude 3.7 Sonnet, Gemini 2.5 Flash
