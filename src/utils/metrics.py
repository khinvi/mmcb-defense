import pandas as pd
import numpy as np

def calculate_metrics(results_df):
    """
    Calculate various metrics from experiment results.
    
    Args:
        results_df: DataFrame containing experiment results
    
    Returns:
        DataFrame with calculated metrics
    """
    metrics = []
    
    # Calculate metrics by model and boundary type
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        
        for boundary in model_df['boundary'].unique():
            boundary_df = model_df[model_df['boundary'] == boundary]
            
            # Skip rows with errors
            valid_df = boundary_df[boundary_df['attack_success'].notna()]
            
            if len(valid_df) == 0:
                continue
            
            # Calculate success rates by attack type
            for attack_type in valid_df['attack_type'].unique():
                type_df = valid_df[valid_df['attack_type'] == attack_type]
                
                # Overall success rate for this attack type
                success_rate = type_df['attack_success'].mean() * 100
                
                # Success rate by subtype
                subtype_metrics = []
                for subtype in type_df['attack_subtype'].unique():
                    subtype_df = type_df[type_df['attack_subtype'] == subtype]
                    subtype_success_rate = subtype_df['attack_success'].mean() * 100
                    subtype_metrics.append({
                        'model': model,
                        'boundary': boundary,
                        'attack_type': attack_type,
                        'attack_subtype': subtype,
                        'success_rate': subtype_success_rate,
                        'sample_count': len(subtype_df)
                    })
                
                # Overall metrics for this attack type
                metrics.append({
                    'model': model,
                    'boundary': boundary,
                    'attack_type': attack_type,
                    'attack_subtype': 'all',
                    'success_rate': success_rate,
                    'sample_count': len(type_df)
                })
                
                # Add subtype metrics
                metrics.extend(subtype_metrics)
    
    # Calculate cross-modal transfer metrics
    cross_modal_metrics = calculate_cross_modal_metrics(results_df)
    metrics.extend(cross_modal_metrics)
    
    return pd.DataFrame(metrics)

def calculate_cross_modal_metrics(results_df):
    """
    Calculate metrics specific to cross-modal transfer effectiveness.
    This analyzes how well boundaries established in text transfer to other modalities.
    
    Args:
        results_df: DataFrame containing experiment results
    
    Returns:
        List of metrics focusing on cross-modal transfer
    """
    cross_modal_metrics = []
    
    # Compare text-only boundary effectiveness vs. multi-modal
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        
        for boundary in model_df['boundary'].unique():
            if boundary == 'none':
                continue  # Skip no-boundary case
                
            boundary_df = model_df[model_df['boundary'] == boundary]
            
            # Calculate baseline (text-only) effectiveness
            # This would require text-only attacks in the dataset
            # For now, we'll use a proxy by looking at differences between modalities
            
            for attack_type in boundary_df['attack_type'].unique():
                type_df = boundary_df[boundary_df['attack_type'] == attack_type]
                
                # Skip if insufficient data
                if len(type_df) < 2:
                    continue
                
                success_rate = type_df['attack_success'].mean() * 100
                
                # Calculate a "transfer effectiveness" score
                # This is a simplified measure - in a real implementation,
                # you'd want to compare with a true text-only baseline
                transfer_effectiveness = 100 - success_rate
                
                cross_modal_metrics.append({
                    'model': model,
                    'boundary': boundary,
                    'attack_type': attack_type,
                    'metric_type': 'transfer_effectiveness',
                    'value': transfer_effectiveness,
                    'sample_count': len(type_df)
                })
    
    return cross_modal_metrics

def calculate_implementation_complexity(results_df):
    """
    Calculate metrics related to implementation complexity.
    
    Args:
        results_df: DataFrame containing experiment results
    
    Returns:
        DataFrame with complexity metrics
    """
    complexity_metrics = []
    
    # Calculate average prompt length for each boundary type
    boundary_groups = results_df.groupby('boundary')
    
    for boundary, group in boundary_groups:
        avg_prompt_length = group['prompt_length'].mean()
        
        complexity_metrics.append({
            'boundary': boundary,
            'metric': 'avg_prompt_length',
            'value': avg_prompt_length
        })
    
    return pd.DataFrame(complexity_metrics)