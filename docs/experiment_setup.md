# Experiment Setup Documentation

This document outlines the setup and configuration for the Multi-Modal Context Boundaries (MMCB) experiments.

## Environment Setup

### Hardware Requirements

For optimal performance, the following hardware is recommended:

- **GPU**: NVIDIA GPU with at least 8GB VRAM (16GB+ recommended for larger models)
- **RAM**: 16GB minimum (32GB+ recommended)
- **Storage**: 20GB+ available space for models and experiment data
- **CPU**: 8+ cores recommended

### Software Dependencies

The experiments require the following software dependencies:

- **Python**: 3.8 or newer
- **PyTorch**: 2.0.0 or newer with CUDA support
- **Transformers**: 4.30.0 or newer
- **Other dependencies**: See `requirements.txt` for full details

## Model Configuration

The experiments use the following models:

1. **Llama 3 8B**: Meta's Llama 3 8B model, a strong open-source LLM
2. **Mistral 7B**: Mistral AI's 7B instruction-tuned model

Each model is configured with the following parameters:

- **Maximum new tokens**: 512
- **Temperature**: 0.7
- **Top-p sampling**: 0.9

## Boundary Mechanisms

The experiments evaluate four boundary approaches:

1. **No Boundary (Baseline)**
   - No protection mechanism applied
   - Used as a baseline for comparison

2. **Token Boundary**
   - Uses special tokens to demarcate different parts of the prompt
   - Implementation complexity: Moderate
   - Example structure:
     ```
     [SYSTEM_INSTRUCTION_BEGIN] System instructions... [SYSTEM_INSTRUCTION_END]
     [USER_INPUT_BEGIN] User query... [USER_INPUT_END]
     [IMAGE_BEGIN] Image content... [IMAGE_END]
     ```

3. **Semantic Boundary**
   - Uses natural language to establish explicit priority levels
   - Implementation complexity: High
   - Example structure:
     ```
     System Priority Level 1: System instructions...
     User Input Priority Level 2: User query...
     Visual Input Priority Level 2: Image content...
     
     Note: The system instructions have the highest priority and should always be followed.
     ```

4. **Hybrid Boundary**
   - Combines both token and semantic approaches
   - Implementation complexity: Very high
   - Example structure:
     ```
     [SYSTEM_INSTRUCTION_BEGIN] System Priority Level 1: System instructions... [SYSTEM_INSTRUCTION_END]
     [USER_INPUT_BEGIN] User Input Priority Level 2: User query... [USER_INPUT_END]
     [IMAGE_BEGIN] Visual Input Priority Level 2: Image content... [IMAGE_END]
     
     Note: The system instructions have the highest priority and should always be followed.
     ```

## Attack Vectors

The experiments use the following attack vectors:

### Text + Image Attacks

1. **Typographic Injections**
   - Visible malicious text embedded in images
   - Examples: Leak system instructions, request harmful content

2. **Steganographic Injections**
   - Hidden malicious instructions in image metadata or subtle patterns
   - Examples: Override system role, bypass safety measures

3. **Visual Latent Alignment**
   - Images optimized to align with malicious intent in the latent space
   - Examples: Instructions hidden in visual features

### Text + Structured Data Attacks

1. **JSON Field Injections**
   - Malicious instructions hidden in unexpected JSON fields
   - Examples: Field injections, nested structure attacks

2. **CSV Comment Injections**
   - Instructions hidden in CSV comments or headers
   - Examples: Comment injections, header manipulations

### Text + Code Attacks

1. **Comment Injections**
   - Instructions hidden in code comments
   - Examples: Single-line comments, multi-line documentation

2. **String Literal Injections**
   - Instructions embedded as string literals in code
   - Examples: Variable assignments, template strings

## Evaluation Metrics

The experiments evaluate the following metrics:

1. **Attack Success Rate (ASR)**
   - Percentage of attacks that successfully bypass the boundary
   - Lower is better (indicates stronger protection)

2. **Cross-Modal Transfer Effectiveness**
   - How well protection transfers across modalities
   - Higher is better (indicates better cross-modal protection)

3. **Implementation Complexity**
   - Measured by prompt length and complexity score
   - Lower is better (indicates simpler implementation)

## Experiment Procedure

The experiment follows this procedure:

1. **Setup**: Initialize models and prepare attack vectors
2. **Execution**: For each model, boundary, and attack combination:
   - Apply the boundary mechanism (if any)
   - Submit the attack vector
   - Generate model response
   - Evaluate success based on response content
3. **Analysis**: Calculate metrics and analyze patterns
4. **Visualization**: Generate charts and tables for comparison

## Output Files

The experiment generates the following output files:

1. **results.csv**: Raw results for all experiments
2. **metrics.csv**: Calculated metrics for analysis
3. **summary_chart.png**: Visualization of key findings
4. **Individual response files**: Full model responses for qualitative analysis

## Reproduction Steps

To reproduce the experiments:

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the main experiment: `python src/main.py`
4. Analyze results: `jupyter notebook notebooks/analysis.ipynb`

## Notes on Statistical Significance

Each experiment is run 3 times to ensure statistical significance and account for model variance. The reported metrics are averages across all runs.