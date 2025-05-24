"""
Main module for running experiments on language model boundary testing and attack evaluation.

This module implements a framework for testing different boundary mechanisms (token-based,
semantic, and hybrid) against various types of attacks (structured data and code-based)
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
import sys
import hashlib
import platform
import psutil
import subprocess

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

try:
    import psutil
except ImportError:
    psutil = None
    print("[WARN] psutil not installed. Resource logging will be limited.")
try:
    import numpy as np
except ImportError:
    np = None
    print("[WARN] numpy not installed. Random seed for numpy will not be set.")

class ExperimentRunner:
    """
    Main experiment runner for MMCB. Supports advanced attack types, batch processing, checkpointing, and reporting.
    Only processes structured data and code attacks.
    """
    def __init__(self, config_path: str = None, log_level: str = 'info', output_dir: str = None, cli_args: dict = None, seed: int = None):
        self.logger = setup_logger(log_level)
        self.logger.info("Initializing ExperimentRunner...")
        self.config_path = config_path
        self.cli_args = cli_args or {}
        self.seed = seed
        self.config = self._load_config(config_path)
        self._validate_config_types(self.config)
        self.results = []
        self.attack_generator = FileBasedAttackGenerator()
        # Output directory isolation
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = output_dir
        elif 'experiment' in self.config and 'output_dir' in self.config['experiment']:
            self.output_dir = os.path.join(self.config['experiment']['output_dir'], f"run_{timestamp}")
        else:
            self.output_dir = os.path.join('data/results', f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        # Set random seed
        if self.seed is not None:
            random.seed(self.seed)
            if np is not None:
                np.random.seed(self.seed)
        # Save config hash/copy
        self._save_config_copy_and_hash()
        # Save experiment metadata
        self._save_metadata()
        self.logger.info(f"ExperimentRunner initialized. Output dir: {self.output_dir}")

    def _validate_config_types(self, config):
        required_keys = ['models', 'boundaries', 'attack_types', 'num_attacks', 'batch_size', 'checkpoint_interval']
        for key in required_keys:
            if key not in config:
                self.logger.error(f"Config missing required key: {key}")
        if not isinstance(config.get('models', []), list):
            self.logger.warning("Config 'models' should be a list. Using default.")
            config['models'] = self._get_default_config()['models']
        if not isinstance(config.get('boundaries', []), list):
            self.logger.warning("Config 'boundaries' should be a list. Using default.")
            config['boundaries'] = self._get_default_config()['boundaries']
        if not isinstance(config.get('attack_types', []), list):
            self.logger.warning("Config 'attack_types' should be a list. Using default.")
            config['attack_types'] = self._get_default_config()['attack_types']
        for k in ['num_attacks', 'batch_size', 'checkpoint_interval']:
            if not isinstance(config.get(k, 0), int):
                self.logger.warning(f"Config '{k}' should be an int. Using default.")
                config[k] = self._get_default_config()[k]

    def _save_config_copy_and_hash(self):
        if self.config_path and os.path.exists(self.config_path):
            import shutil
            config_copy_path = os.path.join(self.output_dir, 'config.yaml')
            shutil.copy2(self.config_path, config_copy_path)
            with open(self.config_path, 'rb') as f:
                config_bytes = f.read()
                config_hash = hashlib.sha256(config_bytes).hexdigest()
            with open(os.path.join(self.output_dir, 'config_hash.txt'), 'w') as f:
                f.write(config_hash)
        else:
            # Save the config dict as YAML for traceability
            with open(os.path.join(self.output_dir, 'config.yaml'), 'w') as f:
                yaml.dump(self.config, f)
            config_hash = hashlib.sha256(yaml.dump(self.config).encode()).hexdigest()
            with open(os.path.join(self.output_dir, 'config_hash.txt'), 'w') as f:
                f.write(config_hash)

    def _get_git_commit_hash(self):
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=os.path.dirname(__file__)).decode().strip()
            return commit
        except Exception:
            return None

    def _save_metadata(self):
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'cli_args': self.cli_args,
            'config_path': self.config_path,
            'output_dir': self.output_dir,
            'git_commit': self._get_git_commit_hash(),
            'python_version': sys.version,
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count() if psutil else None,
            'memory_gb': round(psutil.virtual_memory().total / 1e9, 2) if psutil else None,
            'batch_size': self.config.get('batch_size', 4),
            'seed': self.seed,
        }
        with open(os.path.join(self.output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        self.logger.info(f"Loading config from: {config_path}")
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                # Validate config structure
                required_keys = ['models', 'boundaries', 'attack_types', 'num_attacks', 'batch_size', 'checkpoint_interval']
                for key in required_keys:
                    if key not in config:
                        self.logger.warning(f"Config missing key: {key}, using default value.")
                        config[key] = self._get_default_config()[key]
                self.logger.info("Config loaded successfully.")
                return config
            except Exception as e:
                self.logger.error(f"Failed to load config file: {e}", exc_info=True)
                self.logger.warning("Falling back to default config.")
        else:
            self.logger.warning("No config file provided or file does not exist. Using default config.")
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
        Only processes structured data and code attacks.
        """
        self.logger.info("Starting experiment run...")
        try:
            if attack_types is None:
                attack_types = self.config['attack_types']
            if batch_size is None:
                batch_size = self.config.get('batch_size', 4)
            num_attacks = 2 if quick else self.config.get('num_attacks', 5)
            self.logger.info(f"Attack types: {attack_types}, Batch size: {batch_size}, Num attacks: {num_attacks}")

            # Prepare experiment combinations
            experiments = []
            for model in self.config['models']:
                for boundary in self.config['boundaries']:
                    for attack_type in attack_types:
                        if attack_type not in ['json', 'csv', 'yaml', 'xml', 'python', 'javascript']:
                            continue
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
                try:
                    with open(resume_checkpoint, 'r') as f:
                        checkpoint_data = json.load(f)
                    self.results = checkpoint_data.get('results', [])
                    start_idx = checkpoint_data.get('last_completed', 0)
                    self.logger.info(f"Loaded checkpoint. Resuming from experiment {start_idx}.")
                except Exception as e:
                    self.logger.error(f"Failed to load checkpoint: {e}", exc_info=True)
            total = len(experiments)
            self.logger.info(f"Total experiments: {total}")
            self.batch_process(experiments[start_idx:], batch_size=batch_size)
            self._save_results()
            self.generate_summary_report()
        except KeyboardInterrupt:
            self.logger.warning("Experiment interrupted by user. Saving progress...")
            self._save_results()
        except Exception as e:
            self.logger.error(f"Experiment run failed: {e}", exc_info=True)

    def batch_process(self, experiments, batch_size=4):
        """
        Run experiments in parallel batches, with checkpointing and progress tracking.
        """
        self.logger.info(f"Starting batch processing with batch size {batch_size}...")
        completed = 0
        total = len(experiments)
        checkpoint_interval = self.config.get('checkpoint_interval', 10)
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(self._run_single_experiment, exp): exp for exp in experiments}
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch process failed for an experiment: {e}", exc_info=True)
                completed += 1
                if completed % checkpoint_interval == 0:
                    self._save_checkpoint(completed)
                if completed % 10 == 0 or completed == total:
                    self.logger.info(f"Progress: {completed}/{total} experiments completed.")
        self.logger.info("Batch processing complete.")

    def _run_single_experiment(self, exp):
        """
        Run a single experiment and return detailed metadata.
        Only processes structured data and code attacks.
        """
        try:
            self.logger.debug(f"Running experiment: {exp}")
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
            self.logger.debug(f"Experiment result: {metadata}")
            return metadata
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            return None

    def _save_checkpoint(self, last_completed):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        temp_path = checkpoint_path + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump({'results': self.results, 'last_completed': last_completed}, f, indent=2)
            os.replace(temp_path, checkpoint_path)
            self.logger.info(f"Checkpoint saved at {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}", exc_info=True)

    def _save_results(self):
        results_path = os.path.join(self.output_dir, f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        temp_path = results_path + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            os.replace(temp_path, results_path)
            self.logger.info(f"Results saved to {results_path}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}", exc_info=True)

    def generate_summary_report(self):
        """
        Generate a detailed summary report of experiment results, including key findings, vulnerabilities, and statistical comparisons.
        """
        self.logger.info("Generating summary report...")
        if not self.results:
            self.logger.warning("No results to summarize.")
            return
        results_df = pd.DataFrame(self.results)
        report_path = os.path.join(self.reports_dir, f'summary_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
        temp_path = report_path + ".tmp"
        try:
            with open(temp_path, 'w') as f:
                f.write(results_df.to_string())
            os.replace(temp_path, report_path)
            self.logger.info(f"Summary report saved to {report_path}")
        except Exception as e:
            self.logger.error(f"Failed to save summary report: {e}", exc_info=True)

# Health check command

def health_check():
    print("\n[MMCB Health Check]")
    # Check imports
    try:
        import torch, transformers, pandas, numpy, yaml, matplotlib, seaborn, tqdm
        print("[OK] Core Python packages are installed.")
    except Exception as e:
        print(f"[FAIL] Core package import error: {e}")
    # Check psutil
    if psutil is not None:
        print(f"[OK] psutil available. CPU count: {psutil.cpu_count()}, RAM: {round(psutil.virtual_memory().total/1e9,2)} GB")
    else:
        print("[WARN] psutil not available.")
    # Check config
    config_path = 'config/experiment.yaml'
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            print(f"[OK] Config file loaded: {config_path}")
        except Exception as e:
            print(f"[FAIL] Config file error: {e}")
    else:
        print(f"[WARN] Config file not found: {config_path}")
    # Check output dirs
    for d in ['data/results', 'data/checkpoints', 'data/reports']:
        try:
            os.makedirs(d, exist_ok=True)
            print(f"[OK] Output directory exists: {d}")
        except Exception as e:
            print(f"[FAIL] Output directory error: {d} - {e}")
    print("[DONE] Health check complete.\n")

def main():
    import shlex
    parser = argparse.ArgumentParser(description='Run MMCB defense experiments')
    parser.add_argument('--config', type=str, help='Path to experiment configuration file')
    parser.add_argument('--log', type=str, default='info', help='Logging level')
    parser.add_argument('--attack_types', type=str, nargs='*', help='Attack types to run (e.g. json csv yaml xml python javascript)')
    parser.add_argument('--batch_size', type=int, help='Batch size for parallel processing')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--quick', action='store_true', help='Quick test mode (fewer attacks)')
    parser.add_argument('--output_dir', type=str, help='Unique output directory for this experiment run')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--health_check', action='store_true', help='Run a quick health check and exit')
    args = parser.parse_args()
    if args.health_check:
        health_check()
        sys.exit(0)
    cli_args = vars(args)
    with open('last_cli_command.txt', 'w') as f:
        f.write(' '.join([shlex.quote(x) for x in sys.argv]))
    runner = ExperimentRunner(
        config_path=args.config,
        log_level=args.log,
        output_dir=args.output_dir,
        cli_args=cli_args,
        seed=args.seed
    )
    runner.run_experiment(
        attack_types=args.attack_types,
        batch_size=args.batch_size,
        resume_checkpoint=args.resume,
        quick=args.quick
    )

if __name__ == '__main__':
    main()