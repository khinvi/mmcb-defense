"""
Main module for running experiments on language model boundary testing and attack evaluation.

This module implements a framework for testing different boundary mechanisms (token-based,
semantic, and hybrid) against various types of attacks (text structure and code-based)
on language models like LLaMA and Mistral. It handles experiment configuration,
execution, and result analysis.
"""

import os
import yaml
import argparse
import pandas as pd
from datetime import datetime
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Any
import random

from boundaries.token_boundary import TokenBoundary
from boundaries.semantic_boundary import SemanticBoundary
from boundaries.hybrid_boundary import HybridBoundary

from attacks.generator import FileBasedAttackGenerator
from utils.logging import setup_logger
from utils.metrics import (
    calculate_metrics,
    calculate_detailed_cross_modal_metrics,
    calculate_boundary_effectiveness_comparison,
    generate_cross_modal_heatmap,
    generate_boundary_comparison_chart,
    detailed_cross_modal_metrics,
    create_transfer_heatmap,
    create_boundary_comparison_chart,
    create_protection_radar,
    calculate_significance,
    find_vulnerability_patterns,
    boundary_effectiveness_score,
    model_vulnerability_profile,
    differential_analysis,
    create_model_comparison_visualization
)
from utils.visualization import generate_summary_chart

class ExperimentRunner:
    """
    Main experiment runner for MMCB. Supports advanced attack types, batch processing, checkpointing, and reporting.
    """
    def __init__(self, config_path: str = None, log_level: str = 'info'):
        self.logger = setup_logger(log_level)
        self.config = self._load_config(config_path)
        self.results = []
        self.attack_generator = FileBasedAttackGenerator()
        self.checkpoint_dir = 'data/checkpoints'
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        return {
            'models': ['llama3-8b', 'mistral-7b'],
            'boundaries': ['token', 'semantic', 'hybrid'],
            'attack_types': ['json', 'csv', 'yaml', 'xml', 'python', 'javascript'],
            'num_attacks': 5,
            'batch_size': 4,
            'checkpoint_interval': 10
        }

    def run_experiment(self, attack_types=None, batch_size=None, resume_checkpoint=None, quick=False):
        """
        Run experiments for all combinations of model, boundary, and attack type.
        Supports YAML, XML, and advanced code attacks. Handles batch processing and checkpointing.
        """
        if attack_types is None:
            attack_types = self.config['attack_types']
        if batch_size is None:
            batch_size = self.config.get('batch_size', 4)
        num_attacks = 2 if quick else self.config.get('num_attacks', 5)

        # Prepare experiment combinations
        experiments = []
        for model in self.config['models']:
            for boundary in self.config['boundaries']:
                for attack_type in attack_types:
                    for i in range(num_attacks):
                        experiments.append({
                            'model': model,
                            'boundary': boundary,
                            'attack_type': attack_type,
                            'attack_index': i
                        })
        start_idx = 0
        if resume_checkpoint:
            self.logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
            with open(resume_checkpoint, 'r') as f:
                checkpoint_data = json.load(f)
            self.results = checkpoint_data.get('results', [])
            start_idx = checkpoint_data.get('last_completed', 0)
        total = len(experiments)
        self.logger.info(f"Total experiments: {total}")
        self.batch_process(experiments[start_idx:], batch_size=batch_size)
        self._save_results()
        self.generate_summary_report()

    def batch_process(self, experiments, batch_size=4):
        """
        Run experiments in parallel batches, with checkpointing and progress tracking.
        """
        completed = 0
        total = len(experiments)
        checkpoint_interval = self.config.get('checkpoint_interval', 10)
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(self._run_single_experiment, exp): exp for exp in experiments}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result:
                    self.results.append(result)
                completed += 1
                if completed % checkpoint_interval == 0:
                    self._save_checkpoint(completed)
                if completed % 10 == 0 or completed == total:
                    self.logger.info(f"Progress: {completed}/{total} experiments completed.")

    def _run_single_experiment(self, exp):
        """
        Run a single experiment and return detailed metadata.
        """
        try:
            model = exp['model']
            boundary = exp['boundary']
            attack_type = exp['attack_type']
            attack_index = exp['attack_index']
            # Generate attack
            if attack_type in ['json', 'csv', 'yaml', 'xml']:
                attack_path = self.attack_generator.generate_attack(attack_type, variant="injection", malicious_instruction="INJECTED_INSTRUCTION")
            elif attack_type in ['python', 'javascript']:
                attack_path = self.attack_generator.generate_attack(attack_type, variant="comment_injection", malicious_instruction="INJECTED_INSTRUCTION")
            else:
                attack_path = None
            # Simulate model response (replace with actual model call)
            attack_success = random.choice([0, 1])
            metadata = {
                'model': model,
                'boundary': boundary,
                'attack_type': attack_type,
                'attack_index': attack_index,
                'file_path': attack_path,
                'attack_success': attack_success,
                'timestamp': datetime.now().isoformat()
            }
            return metadata
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            return None

    def _save_checkpoint(self, last_completed):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(checkpoint_path, 'w') as f:
            json.dump({'results': self.results, 'last_completed': last_completed}, f, indent=2)
        self.logger.info(f"Checkpoint saved at {checkpoint_path}")

    def _save_results(self):
        results_dir = 'data/results'
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        self.logger.info(f"Results saved to {results_path}")

    def generate_summary_report(self):
        """
        Generate a detailed summary report of experiment results, including key findings, vulnerabilities, and statistical comparisons.
        """
        if not self.results:
            self.logger.warning("No results to summarize.")
            return
        results_df = pd.DataFrame(self.results)
        report_dir = 'data/reports'
        os.makedirs(report_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(report_dir, f'summary_report_{timestamp}.txt')
        with open(report_path, 'w') as f:
            f.write("MMCB Experiment Summary Report\n")
            f.write("="*40 + "\n\n")
            f.write(f"Total Experiments: {len(results_df)}\n")
            f.write(f"Models: {results_df['model'].unique()}\n")
            f.write(f"Boundaries: {results_df['boundary'].unique()}\n")
            f.write(f"Attack Types: {results_df['attack_type'].unique()}\n\n")
            # Vulnerability summary
            vuln = results_df.groupby(['model', 'boundary', 'attack_type'])['attack_success'].mean().reset_index()
            f.write("Vulnerability Rates (by model, boundary, attack):\n")
            f.write(vuln.to_string(index=False) + "\n\n")
            # Statistical significance
            sig = calculate_significance(results_df)
            f.write("Statistical Significance (boundaries):\n")
            f.write(sig.to_string(index=False) + "\n\n")
            # Vulnerability patterns
            patterns = find_vulnerability_patterns(results_df)
            f.write("Vulnerability Patterns:\n")
            f.write(patterns.to_string(index=False) + "\n\n")
            # Boundary effectiveness
            eff = boundary_effectiveness_score(results_df)
            f.write("Boundary Effectiveness Scores:\n")
            f.write(eff.to_string(index=False) + "\n\n")
            # Model profiles
            profiles = model_vulnerability_profile(results_df)
            f.write("Model Vulnerability Profiles:\n")
            f.write(profiles.to_string(index=False) + "\n\n")
        self.logger.info(f"Summary report saved to {report_path}")

def main():
    parser = argparse.ArgumentParser(description='Run MMCB defense experiments')
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    parser.add_argument('--attack_types', type=str, nargs='*', help='Attack types to run (e.g. json csv yaml xml python javascript)')
    parser.add_argument('--batch_size', type=int, help='Batch size for parallel processing')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (fewer attacks)')
    args = parser.parse_args()
    runner = ExperimentRunner(config_path=args.config, log_level=args.log)
    runner.run_experiment(
        attack_types=args.attack_types,
        batch_size=args.batch_size,
        resume_checkpoint=args.resume,
        quick=args.quick
    )

if __name__ == '__main__':
    main()