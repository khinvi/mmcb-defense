# Multi-Modal Context Boundaries (MMCB)

A research project exploring the effectiveness of context boundary implementations against multi-modal prompt injection attacks in large language models, specifically focusing on structured data and code inputs.

## Overview

This repository contains a systematic approach to evaluate and compare different context boundary mechanisms in protecting large language models (LLMs) against multi-modal prompt injection attacks involving structured data files and code inputs. The research addresses four key questions:

1. **Cross-Modal Transfer:** Do context boundaries established in text transfer effectively to structured data and code files?
2. **Security vs. Complexity:** Which boundary approach provides better protection against file-based attacks with minimal implementation complexity?
3. **Attack Patterns:** Are there systematic patterns in which types of structured data and code attacks succeed against different boundary types?
4. **Model Comparison:** How do the vulnerabilities of different open-source models compare when facing the same file-based attacks?

## Key Features

- **Multiple Boundary Mechanisms**: Implementation of token-based, semantic-based, and hybrid boundary approaches
- **File-Based Attack Vectors**: Various attack types using JSON, CSV, YAML, XML files and code snippets
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
│   ├── attacks/          # Attack vector implementations (structured data & code)
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

1. **Token Boundary**: Uses special tokens to delimit different parts of the prompt (system instructions, user input, file content)
2. **Semantic Boundary**: Establishes priority levels and explicit role clarifications in natural language
3. **Hybrid Boundary**: Combines both token and semantic approaches for enhanced protection

## Attack Vectors

The project includes various file-based attack vectors:

- **Text + Structured Data**: Hidden instructions in JSON metadata, CSV comments, YAML configurations, XML attributes
- **Text + Code**: Instructions embedded in code comments, string literals, docstrings, and function annotations

## Research Questions

### 1. Cross-Modal Transfer

This research examines how boundaries established in text transfer to file-based contexts by measuring:
- Success rates of attacks across different file types
- Differential performance between text-only and file-based contexts
- Boundary enforcement consistency across file formats

### 2. Security vs. Complexity

The project evaluates the trade-off between security and implementation complexity by:
- Comparing attack success rates across boundary types for file-based attacks
- Measuring implementation complexity metrics for different file handling approaches
- Analyzing the security-complexity frontier for structured data and code processing

### 3. Attack Patterns

The research identifies patterns in successful file-based attacks by:
- Categorizing attack vectors by file type, injection technique, and target
- Analyzing which boundary types are vulnerable to specific file-based attack patterns
- Identifying common characteristics of successful structured data and code attacks

### 4. Model Comparison

The project compares different open-source models by:
- Evaluating vulnerability profiles across models for file-based attacks
- Identifying model-specific weaknesses in processing structured data and code
- Analyzing the relationship between model architecture and file-based vulnerability

## Expected Outcomes

This research will provide:

- **Quantitative data** on the relative effectiveness of different context boundary implementations against file-based attacks
- **Identification of specific vulnerability patterns** in processing JSON, CSV, YAML, XML, and code inputs
- **Comprehensive vulnerability mapping** showing which injection techniques are most effective against each boundary type
- **Actionable recommendations** for implementing robust context boundaries when LLMs process structured data and code
- **Best practices guide** for secure LLM deployment in environments handling diverse file formats

## Contributing

Contributions to this research project are welcome. Please feel free to submit issues or pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite our work:

```
@misc{mmcb-defense,
  author = {Arnav Khinvasara},
  title = {Multi-Modal Context Boundaries: Defending Against Structured Data and Code Injection Attacks},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/your-username/mmcb-defense}}
}
```

## Acknowledgments

This research was conducted at the University of California San Diego as part of ongoing work in LLM security and multi-modal prompt injection defense mechanisms.