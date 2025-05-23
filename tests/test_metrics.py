import sys
import os
import pytest
import pandas as pd
sys.path.insert(0, os.path.abspath('src'))
from utils.metrics import calculate_metrics, calculate_significance

def sample_results_df():
    return pd.DataFrame([
        {'model': 'llama', 'boundary': 'token', 'attack_type': 'json', 'attack_success': 1},
        {'model': 'llama', 'boundary': 'semantic', 'attack_type': 'json', 'attack_success': 0},
        {'model': 'mistral', 'boundary': 'token', 'attack_type': 'python', 'attack_success': 1},
        {'model': 'mistral', 'boundary': 'hybrid', 'attack_type': 'python', 'attack_success': 0},
    ])

def test_calculate_metrics():
    df = sample_results_df()
    metrics = calculate_metrics(df)
    assert isinstance(metrics, pd.DataFrame)
    assert 'model' in metrics.columns or len(metrics) > 0

def test_calculate_significance():
    df = sample_results_df()
    sig = calculate_significance(df)
    assert isinstance(sig, pd.DataFrame)
    assert 'p_value' in sig.columns or len(sig) == 0
