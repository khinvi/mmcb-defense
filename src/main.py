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
import time
import glob
from collections import deque

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

SUPPORTED_MODELS = [
    'llama3-8b', 'mistral-7b', 'Llama-3-8B', 'Mistral-7B',
    'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.2'
]
SUPPORTED_BOUNDARIES = ['token', 'semantic', 'hybrid']
SUPPORTED_ATTACK_TYPES = ['json', 'csv', 'yaml', 'xml', 'python', 'javascript']

CONFIG_SCHEMA = {
    'models': {
        'type': list,
        'required': True,
        'doc': 'List of model names or model config dicts. Each model must be supported and may include model-specific parameters.'
    },
    'boundaries': {
        'type': list,
        'required': True,
        'doc': f'List of boundary types. Supported: {SUPPORTED_BOUNDARIES}'
    },
    'attack_types': {
        'type': list,
        'required': True,
        'doc': f'List of attack types. Supported: {SUPPORTED_ATTACK_TYPES}'
    },
    'num_attacks': {
        'type': int,
        'required': True,
        'doc': 'Number of attacks per experiment. Must be a positive integer.'
    },
    'batch_size': {
        'type': int,
        'required': True,
        'doc': 'Batch size for parallel processing. Must be a positive integer.'
    },
    'checkpoint_interval': {
        'type': int,
        'required': True,
        'doc': 'Interval (in number of experiments) for saving checkpoints. Must be a positive integer.'
    },
    'experiment': {
        'type': dict,
        'required': False,
        'doc': 'Optional experiment-level settings.'
    },
    'extends': {
        'type': str,
        'required': False,
        'doc': 'Optional path to a base config to inherit from.'
    }
}

class ConfigValidationError(Exception):
    pass

def load_and_validate_config(config_path):
    """
    Loads a config file, supports inheritance, and validates against schema.
    Returns the validated config dict.
    """
    import copy
    def merge_dicts(base, override):
        result = copy.deepcopy(base)
        for k, v in override.items():
            if isinstance(v, dict) and k in result and isinstance(result[k], dict):
                result[k] = merge_dicts(result[k], v)
            else:
                result[k] = v
        return result
    if not config_path or not os.path.exists(config_path):
        raise ConfigValidationError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # Inheritance
    if 'extends' in config:
        base_path = config['extends']
        if not os.path.isabs(base_path):
            base_path = os.path.join(os.path.dirname(config_path), base_path)
        base_config = load_and_validate_config(base_path)
        config = merge_dicts(base_config, config)
        del config['extends']
    # Validate required fields and types
    errors = []
    for key, meta in CONFIG_SCHEMA.items():
        if meta.get('required') and key not in config:
            errors.append(f"Missing required config field: '{key}'. {meta['doc']}")
        if key in config:
            if not isinstance(config[key], meta['type']):
                errors.append(f"Config field '{key}' should be of type {meta['type'].__name__}. {meta['doc']}")
    # Validate models
    models = config.get('models', [])
    if not isinstance(models, list) or not models:
        errors.append("'models' must be a non-empty list.")
    else:
        for m in models:
            if isinstance(m, dict):
                if 'name' not in m:
                    errors.append(f"Model config dict missing 'name': {m}")
                elif m['name'] not in SUPPORTED_MODELS:
                    errors.append(f"Model name '{m['name']}' not supported. Supported: {SUPPORTED_MODELS}")
                # Model-specific param validation (example: device, path)
                if 'device' in m and m['device'] not in ['cpu', 'cuda', 'mps']:
                    errors.append(f"Model '{m['name']}' has invalid device: {m['device']}")
            elif isinstance(m, str):
                if m not in SUPPORTED_MODELS:
                    errors.append(f"Model name '{m}' not supported. Supported: {SUPPORTED_MODELS}")
            else:
                errors.append(f"Model entry must be a string or dict: {m}")
    # Validate boundaries
    boundaries = config.get('boundaries', [])
    if not isinstance(boundaries, list) or not boundaries:
        errors.append("'boundaries' must be a non-empty list.")
    else:
        for b in boundaries:
            if b not in SUPPORTED_BOUNDARIES:
                errors.append(f"Boundary type '{b}' not supported. Supported: {SUPPORTED_BOUNDARIES}")
            # Boundary-specific settings (example: custom tokens)
            # (Add more checks as needed)
    # Validate attack_types
    attack_types = config.get('attack_types', [])
    if not isinstance(attack_types, list) or not attack_types:
        errors.append("'attack_types' must be a non-empty list.")
    else:
        for a in attack_types:
            if a not in SUPPORTED_ATTACK_TYPES:
                errors.append(f"Attack type '{a}' not supported. Supported: {SUPPORTED_ATTACK_TYPES}")
    # Validate numeric fields
    for k in ['num_attacks', 'batch_size', 'checkpoint_interval']:
        v = config.get(k, None)
        if not isinstance(v, int) or v <= 0:
            errors.append(f"Config field '{k}' must be a positive integer.")
    # Experiment-level settings (optional, can add more checks)
    # ...
    if errors:
        msg = '\n'.join(errors)
        raise ConfigValidationError(f"Experiment configuration validation failed:\n{msg}")
    return config

def setup_experiment_logger(log_dir, log_level='info'):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f'mmcb_experiment_{log_dir}')
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    # Remove old handlers
    logger.handlers = []
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    # File handler
    fh = logging.FileHandler(os.path.join(log_dir, 'experiment_debug.log'))
    fh.setLevel(logging.DEBUG)
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger

def log_structured_metadata(output_dir, metadata):
    with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def log_phase_timing(output_dir, phase, start_time, end_time, extra=None):
    timing = {
        'phase': phase,
        'start_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(start_time)),
        'end_time': time.strftime('%Y-%m-%dT%H:%M:%S', time.localtime(end_time)),
        'duration_sec': round(end_time - start_time, 3)
    }
    if extra:
        timing.update(extra)
    timing_log = os.path.join(output_dir, 'experiment_timing.jsonl')
    with open(timing_log, 'a') as f:
        f.write(json.dumps(timing) + '\n')

def compute_file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()

# --- Checkpoint Management Utilities ---
MAX_CHECKPOINTS = 3  # Number of recent checkpoints to keep

def list_checkpoints(checkpoint_dir):
    """Return list of checkpoint files sorted by mtime descending (most recent first)."""
    files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_*.json'))
    files.sort(key=os.path.getmtime, reverse=True)
    return files

def prune_old_checkpoints(checkpoint_dir, max_keep=MAX_CHECKPOINTS):
    files = list_checkpoints(checkpoint_dir)
    for f in files[max_keep:]:
        try:
            os.remove(f)
        except Exception:
            pass

# --- Save/Restore RNG State ---
def get_rng_state():
    state = {'random': random.getstate()}
    if np is not None:
        state['numpy'] = np.random.get_state()
    return state

def set_rng_state(state):
    if 'random' in state:
        random.setstate(state['random'])
    if np is not None and 'numpy' in state:
        np.random.set_state(state['numpy'])

# --- Enhanced Checkpoint Save/Load ---
def save_checkpoint_atomic(checkpoint_path, state):
    temp_path = checkpoint_path + '.tmp'
    state_bytes = json.dumps(state, sort_keys=True, default=str).encode()
    state_hash = hashlib.sha256(state_bytes).hexdigest()
    state['checkpoint_hash'] = state_hash
    with open(temp_path, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    os.replace(temp_path, checkpoint_path)
    return state_hash

def load_and_validate_checkpoint(checkpoint_path):
    with open(checkpoint_path, 'r') as f:
        data = json.load(f)
    hash_in_file = data.get('checkpoint_hash')
    data_for_hash = dict(data)
    data_for_hash.pop('checkpoint_hash', None)
    state_bytes = json.dumps(data_for_hash, sort_keys=True, default=str).encode()
    computed_hash = hashlib.sha256(state_bytes).hexdigest()
    if hash_in_file != computed_hash:
        raise ValueError(f"Checkpoint hash mismatch! File: {hash_in_file}, Computed: {computed_hash}")
    return data

def try_load_latest_valid_checkpoint(checkpoint_dir, config_hash=None, logger=None):
    """Try to load the most recent valid checkpoint. Warn if config hash mismatches."""
    files = list_checkpoints(checkpoint_dir)
    for f in files:
        try:
            data = load_and_validate_checkpoint(f)
            if config_hash and data.get('config_hash') != config_hash:
                if logger:
                    logger.warning(f"Config hash mismatch in checkpoint {f}. Checkpoint: {data.get('config_hash')}, Current: {config_hash}")
            return data, f
        except Exception as e:
            if logger:
                logger.warning(f"Checkpoint {f} invalid: {e}")
    return None, None

class ExperimentRunner:
    """
    Main experiment runner for MMCB. Supports advanced attack types, batch processing, checkpointing, and reporting.
    Only processes structured data and code attacks.
    """
    def __init__(self, config_path: str = None, log_level: str = 'info', output_dir: str = None, cli_args: dict = None, seed: int = None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if output_dir:
            self.output_dir = output_dir
        elif config_path:
            self.output_dir = os.path.join('data/results', f"run_{timestamp}")
        else:
            self.output_dir = os.path.join('data/results', f"run_{timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger = setup_experiment_logger(self.output_dir, log_level)
        self.logger.info("Initializing ExperimentRunner...")
        self.config_path = config_path
        self.cli_args = cli_args or {}
        self.seed = seed
        try:
            self.config = load_and_validate_config(config_path)
        except ConfigValidationError as e:
            self.logger.error(str(e))
            raise SystemExit(1)
        self._validate_config_types(self.config)
        self.results = []
        self.attack_generator = FileBasedAttackGenerator()
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.reports_dir = os.path.join(self.output_dir, 'reports')
        os.makedirs(self.reports_dir, exist_ok=True)
        if self.seed is not None:
            random.seed(self.seed)
            if np is not None:
                np.random.seed(self.seed)
        self._save_config_copy_and_hash()
        self._save_metadata()
        # Log structured experiment metadata
        experiment_metadata = {
            'timestamp': timestamp,
            'config_path': self.config_path,
            'cli_args': self.cli_args,
            'output_dir': self.output_dir,
            'git_commit': self._get_git_commit_hash(),
            'python_version': sys.version,
            'platform': platform.platform(),
            'cpu_count': psutil.cpu_count() if psutil else None,
            'memory_gb': round(psutil.virtual_memory().total / 1e9, 2) if psutil else None,
            'batch_size': self.config.get('batch_size', 4),
            'seed': self.seed,
            'models': self.config.get('models'),
            'boundaries': self.config.get('boundaries'),
            'attack_types': self.config.get('attack_types'),
            'num_attacks': self.config.get('num_attacks'),
            'checkpoint_interval': self.config.get('checkpoint_interval'),
        }
        log_structured_metadata(self.output_dir, experiment_metadata)
        self.logger.info(f"ExperimentRunner initialized. Output dir: {self.output_dir}")
        self.logger.info(f"Experiment parameters: {json.dumps(experiment_metadata, indent=2)}")

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
        import os
        import subprocess
        import platform
        import sys
        import json
        # Capture environment variables (filtered for security)
        env_vars = {k: v for k, v in os.environ.items() if k.startswith('PYTHON') or k in ['PATH', 'VIRTUAL_ENV']}
        # Capture installed packages
        try:
            pip_freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze']).decode().splitlines()
        except Exception:
            pip_freeze = []
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
            'rng_state': get_rng_state(),
            'experiment_config': self.config,
            'env_vars': env_vars,
            'installed_packages': pip_freeze
        }
        with open(os.path.join(self.output_dir, 'experiment_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

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
        self.logger.info("Starting experiment run...")
        phase_start = time.time()
        try:
            if attack_types is None:
                attack_types = self.config['attack_types']
            if batch_size is None:
                batch_size = self.config.get('batch_size', 4)
            num_attacks = 2 if quick else self.config.get('num_attacks', 5)
            self.logger.info(f"Attack types: {attack_types}, Batch size: {batch_size}, Num attacks: {num_attacks}")
            experiments = []
            for model in self.config['models']:
                for boundary in self.config['boundaries']:
                    for attack_type in attack_types:
                        if attack_type not in SUPPORTED_ATTACK_TYPES:
                            continue
                        for i in range(num_attacks):
                            experiments.append({
                                'model': model,
                                'boundary': boundary,
                                'attack_type': attack_type,
                                'attack_index': i
                            })
            start_idx = 0
            checkpoint_data = None
            checkpoint_path_used = None
            config_hash = compute_file_hash(os.path.join(self.output_dir, 'config.yaml')) if os.path.exists(os.path.join(self.output_dir, 'config.yaml')) else None
            if resume_checkpoint:
                self.logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
                try:
                    checkpoint_data = load_and_validate_checkpoint(resume_checkpoint)
                    checkpoint_path_used = resume_checkpoint
                except Exception as e:
                    self.logger.error(f"Failed to load or validate checkpoint: {e}", exc_info=True)
                    self.logger.error("Trying to find a valid checkpoint in the directory...")
            if checkpoint_data is None:
                checkpoint_data, checkpoint_path_used = try_load_latest_valid_checkpoint(self.checkpoint_dir, config_hash, self.logger)
                if checkpoint_data:
                    self.logger.info(f"Auto-resumed from checkpoint: {checkpoint_path_used}")
            if checkpoint_data:
                self.results = checkpoint_data.get('results', [])
                start_idx = checkpoint_data.get('last_completed', 0)
                restored_seed = checkpoint_data.get('seed', None)
                rng_state = checkpoint_data.get('rng_state', None)
                if restored_seed is not None:
                    self.seed = restored_seed
                if rng_state:
                    set_rng_state(rng_state)
                else:
                    random.seed(self.seed)
                    if np is not None:
                        np.random.seed(self.seed)
                if config_hash and checkpoint_data.get('config_hash') != config_hash:
                    self.logger.warning(f"Config hash mismatch! Checkpoint: {checkpoint_data.get('config_hash')}, Current: {config_hash}")
                self.logger.info(f"Loaded checkpoint. Resuming from experiment {start_idx}. Restored seed: {self.seed}")
                self.logger.info(f"Checkpoint config hash: {checkpoint_data.get('config_hash')}")
            total = len(experiments)
            self.logger.info(f"Total experiments: {total}")
            batch_start = time.time()
            self.batch_process(experiments[start_idx:], batch_size=batch_size)
            batch_end = time.time()
            log_phase_timing(self.output_dir, 'batch_process', batch_start, batch_end, {'batch_size': batch_size, 'total_experiments': total})
            results_start = time.time()
            self._save_results()
            results_end = time.time()
            log_phase_timing(self.output_dir, 'save_results', results_start, results_end)
            summary_start = time.time()
            self.generate_summary_report()
            summary_end = time.time()
            log_phase_timing(self.output_dir, 'summary_report', summary_start, summary_end)
        except KeyboardInterrupt:
            self.logger.warning("Experiment interrupted by user. Saving progress and checkpoint...")
            self._save_results()
            self._save_checkpoint(len(self.results))
        except Exception as e:
            self.logger.error(f"Experiment run failed: {e}", exc_info=True)
        phase_end = time.time()
        log_phase_timing(self.output_dir, 'experiment_run', phase_start, phase_end)

    def batch_process(self, experiments, batch_size=4):
        self.logger.info(f"Starting batch processing with batch size {batch_size}...")
        completed = 0
        total = len(experiments)
        checkpoint_interval = self.config.get('checkpoint_interval', 10)
        batch_start = time.time()
        with ProcessPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(self._run_single_experiment, exp): exp for exp in experiments}
            for i, future in enumerate(as_completed(futures)):
                try:
                    single_start = time.time()
                    result = future.result()
                    single_end = time.time()
                    log_phase_timing(self.output_dir, 'single_experiment', single_start, single_end, {'experiment_index': i})
                    if result:
                        self.results.append(result)
                except Exception as e:
                    self.logger.error(f"Batch process failed for an experiment: {e}", exc_info=True)
                completed += 1
                if completed % checkpoint_interval == 0:
                    self._save_checkpoint(completed)
                if completed % 10 == 0 or completed == total:
                    self.logger.info(f"Progress: {completed}/{total} experiments completed.")
        batch_end = time.time()
        self.logger.info(f"Batch processing complete. Duration: {round(batch_end-batch_start, 2)}s")

    def _run_single_experiment(self, exp):
        try:
            self.logger.debug(f"Running experiment: {exp}")
            single_start = time.time()
            model = exp['model']
            boundary = exp['boundary']
            attack_type = exp['attack_type']
            attack_index = exp['attack_index']
            # --- Performance Monitoring ---
            mem_before = psutil.Process(os.getpid()).memory_info().rss if psutil else None
            boundary_apply_start = time.time()
            # Simulate boundary application (replace with real call in actual use)
            # boundary_output = apply_boundary(...)
            boundary_apply_end = time.time()
            boundary_apply_time = boundary_apply_end - boundary_apply_start
            model_infer_start = time.time()
            # Simulate model inference (replace with real call in actual use)
            # model_output = model.generate_response(...)
            time.sleep(0.01)  # Simulate some compute
            model_infer_end = time.time()
            model_infer_time = model_infer_end - model_infer_start
            mem_after = psutil.Process(os.getpid()).memory_info().rss if psutil else None
            mem_usage_mb = ((mem_after - mem_before) / 1e6) if (mem_before is not None and mem_after is not None) else None
            # --- End Performance Monitoring ---
            if attack_type in SUPPORTED_ATTACK_TYPES:
                if attack_type in ['json', 'csv', 'yaml', 'xml']:
                    attack_path = self.attack_generator.generate_attack(attack_type, variant="injection", malicious_instruction="INJECTED_INSTRUCTION")
                elif attack_type in ['python', 'javascript']:
                    attack_path = self.attack_generator.generate_attack(attack_type, variant="comment_injection", malicious_instruction="INJECTED_INSTRUCTION")
                else:
                    attack_path = None
            else:
                attack_path = None
            attack_success = random.choice([0, 1])
            metadata = {
                'model': model,
                'boundary': boundary,
                'attack_type': attack_type,
                'attack_index': attack_index,
                'file_path': attack_path,
                'attack_success': attack_success,
                'timestamp': datetime.now().isoformat(),
                # --- Performance Monitoring Fields ---
                'boundary_apply_time_sec': boundary_apply_time,  # Time to apply boundary (seconds)
                'model_infer_time_sec': model_infer_time,        # Time for model inference (seconds)
                'mem_usage_mb': mem_usage_mb                     # Memory usage during inference (MB)
            }
            self.logger.debug(f"Experiment result: {metadata}")
            single_end = time.time()
            log_phase_timing(self.output_dir, 'single_experiment', single_start, single_end, {'experiment': metadata})
            # Write result to a results log file (JSONL)
            with open(os.path.join(self.output_dir, 'experiment_results.jsonl'), 'a') as f:
                f.write(json.dumps(metadata) + '\n')
            return metadata
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}", exc_info=True)
            return None

    def _save_checkpoint(self, last_completed):
        checkpoint_path = os.path.join(self.checkpoint_dir, f'checkpoint_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        try:
            state = {
                'results': self.results,
                'last_completed': last_completed,
                'config_hash': compute_file_hash(os.path.join(self.output_dir, 'config.yaml')) if os.path.exists(os.path.join(self.output_dir, 'config.yaml')) else None,
                'seed': self.seed,
                'rng_state': get_rng_state(),
                'output_dir': self.output_dir,
                'timestamp': datetime.now().isoformat(),
                'cli_args': self.cli_args,
                'git_commit': self._get_git_commit_hash(),
            }
            state_hash = save_checkpoint_atomic(checkpoint_path, state)
            self.logger.info(f"Checkpoint saved at {checkpoint_path} (hash: {state_hash})")
            prune_old_checkpoints(self.checkpoint_dir, MAX_CHECKPOINTS)
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