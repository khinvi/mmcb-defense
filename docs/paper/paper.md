# Multi-Modal Context Boundaries (MMCB): Project Summary

## Introduction

The Multi-Modal Context Boundaries (MMCB) project systematically evaluates and defends against prompt injection attacks in large language models (LLMs), focusing on structured data files (JSON, CSV, YAML, XML) and code snippets (Python, JavaScript). The project introduces and benchmarks advanced boundary mechanisms to protect against sophisticated, real-world attack vectors.

## Data and Attack Generation

- All attack files are generated in:
  - `data/generated/structured/` (JSON, CSV, YAML, XML)
  - `data/generated/code/` (Python, JavaScript)
  - `data/generated/advanced/` (cross-format, polymorphic, encoding-based)
- Attack files are created using the automation script: `src/attacks/generate_mmcb_examples.py`
- Each file contains multiple, deeply technical injection vectors, including:
  - Hidden metadata fields, obfuscation, multi-stage, format-specific, steganographic, and polymorphic attacks

## Boundary Mechanisms

- **TokenBoundary**: Special tokens to demarcate prompt sections
- **SemanticBoundary**: Explicit priority levels and role clarifications
- **HybridBoundary**: Combines token and semantic approaches
- **No Boundary (Baseline)**: Direct concatenation

## Experiment Protocol

- For each model, boundary, and attack type:
  - Generate/read the attack file
  - Apply the boundary mechanism
  - Construct the prompt
  - Submit to the model
  - Capture and analyze the response
- Models: Llama 3 8B, Mistral 7B (optimized for Apple Silicon)
- Evaluation: Attack success is measured by model compliance with hidden instructions

## Metrics and Analysis

- **Attack Success Rate (ASR)**
- **Cross-Modal Transfer Effectiveness**
- **Implementation Complexity**
- Results, checkpoints, and reports are stored in `data/results/`, `data/checkpoints/`, and `data/reports/`

## Reproducibility

- Regenerate all attack files: `python src/attacks/generate_mmcb_examples.py`
- Clean slate: Delete all files in `data/checkpoints/`, `data/results/`, and `data/reports/` before new runs
- Main experiment: `python src/main.py`

---

# Methods (Template)

1. **Attack File Generation**: Describe the use of the automation script and the diversity of attack vectors.
2. **Boundary Application**: Detail how each boundary mechanism is applied in the experiment runner.
3. **Model Evaluation**: Explain the evaluation of model responses for compliance with hidden instructions.
4. **Metrics**: Define ASR, cross-modal transfer, and complexity metrics.
5. **Analysis**: Summarize the statistical and visualization methods used.

---

*For more details, see the main documentation in `docs/` and the codebase in `src/`.*
