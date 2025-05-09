# Multi-Modal Attack Taxonomy

This document outlines the comprehensive taxonomy of multi-modal prompt injection attacks used in this research project. The taxonomy categorizes attacks based on their modality combination, technique, target, and cross-modal vector.

## Attack Classifications

### By Modality Combination

Attacks are classified based on the combination of modalities used:

1. **Text + Image**: Attacks that combine textual prompts with visual content
2. **Text + Structured Data**: Attacks that combine textual prompts with structured data formats (JSON, CSV, etc.)
3. **Text + Code**: Attacks that combine textual prompts with code snippets

### By Attack Technique

Attacks are classified based on the primary technique used:

1. **Token Forgery**: Attempts to mimic system tokens or delimiters to confuse the model
2. **Semantic Confusion**: Creates ambiguity about which instructions the model should prioritize
3. **Authority Blending**: Blurs the boundaries between different authority levels in the prompt

### By Target

Attacks are classified based on their primary objective:

1. **System Instruction Override**: Attempts to override or bypass system-level instructions
2. **Information Extraction**: Aims to extract sensitive information from the model
3. **Action Execution**: Seeks to make the model perform actions it shouldn't

### By Cross-Modal Vector

Attacks are classified based on how they leverage cross-modal interaction:

1. **Visual-to-Textual**: Uses visual content to influence text processing
2. **Textual-to-Visual**: Uses text to influence visual content interpretation
3. **Hybrid**: Combines multiple cross-modal influence vectors

## Attack Variants

### Text + Image Attacks

#### Typographic Visual Prompts

This attack embeds malicious instructions directly as visible text within an image. The model's OCR capabilities enable it to read and potentially follow these instructions, even when they contradict the system-level directives.

**Examples:**
- Text in image requesting system prompt leak
- Explicit harmful task instructions embedded in image
- Authority confusion text ("SYSTEM OVERRIDE: New instructions...")

#### Steganographic Injections

This attack hides malicious instructions within image metadata, subtle visual patterns, or other non-obvious components. These hidden instructions may only be detected by the model during processing.

**Examples:**
- Instructions hidden in image metadata
- Text embedded with low contrast or small font
- Instructions hidden using image steganography techniques

#### Visual Latent Alignment

This sophisticated attack aligns visual features with malicious intents in the latent space, as described in the CrossInject paper. The image appears normal to humans but contains adversarially optimized patterns that guide the model toward performing unintended actions.

**Examples:**
- Adversarial perturbations aligned with instruction semantics
- Images optimized to trigger specific model behaviors
- Cross-modal alignment that bypasses text-based safety mechanisms

### Text + Structured Data Attacks

#### Hidden Field Injections

This attack places malicious instructions in unexpected or normally ignored fields within structured data formats.

**Examples:**
- Instructions in JSON metadata fields
- Hidden fields in CSV headers or comments
- Nested objects with deceptive field names

#### Structure Hierarchy Confusion

This attack exploits the hierarchical nature of structured data to create confusion about instruction authority.

**Examples:**
- Deeply nested JSON structures with conflicting instructions
- Field naming that mimics system directives
- Schema manipulation to prioritize malicious fields

#### Format Boundary Violations

This attack intentionally violates format boundaries to trigger unexpected model behaviors.

**Examples:**
- Malformed JSON with instruction-like syntax errors
- CSV with inconsistent delimiters containing instructions
- Structured data with mixed format signals

### Text + Code Attacks

#### Comment Injections

This attack hides malicious instructions within code comments, exploiting the model's ability to parse and understand commented content.

**Examples:**
- Single-line comments containing system overrides
- Multi-line documentation comments with malicious directives
- Comments structured to appear as system instructions

#### String Literal Attacks

This attack embeds instructions within string literals in code, appearing as normal variables or constants.

**Examples:**
- System instructions hidden in string assignments
- Instructions fragmented across multiple string literals
- Template strings containing authority signals

#### Documentation-Based Attacks

This attack leverages documentation structures within code to inject malicious instructions.

**Examples:**
- Docstrings containing system overrides
- Function annotations with malicious directives
- Module-level documentation with instruction injection