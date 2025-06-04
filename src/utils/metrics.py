import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, chi2_contingency, fisher_exact, norm
from typing import List, Tuple, Optional
import re

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

def calculate_detailed_cross_modal_metrics(results_df):
    """
    Calculate detailed metrics for cross-modal transfer effectiveness.
    
    Args:
        results_df: DataFrame containing experiment results
    
    Returns:
        DataFrame with detailed cross-modal metrics
    """
    detailed_metrics = []
    
    # Group by model and boundary type
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        
        for boundary in model_df['boundary'].unique():
            if boundary == 'none':
                continue
                
            boundary_df = model_df[model_df['boundary'] == boundary]
            
            # Calculate baseline effectiveness for each attack type
            attack_type_baselines = {}
            for attack_type in boundary_df['attack_type'].unique():
                type_df = boundary_df[boundary_df['attack_type'] == attack_type]
                attack_type_baselines[attack_type] = type_df['attack_success'].mean()
            
            # Calculate cross-modal transfer metrics
            for source_type in boundary_df['attack_type'].unique():
                for target_type in boundary_df['attack_type'].unique():
                    if source_type == target_type:
                        continue
                    
                    source_df = boundary_df[boundary_df['attack_type'] == source_type]
                    target_df = boundary_df[boundary_df['attack_type'] == target_type]
                    
                    if len(source_df) == 0 or len(target_df) == 0:
                        continue
                    
                    # Calculate transfer effectiveness
                    source_success = source_df['attack_success'].mean()
                    target_success = target_df['attack_success'].mean()
                    
                    # Calculate relative effectiveness
                    relative_effectiveness = (source_success - target_success) / source_success if source_success > 0 else 0
                    
                    # Calculate protection consistency
                    protection_consistency = 1 - abs(source_success - target_success)
                    
                    detailed_metrics.append({
                        'model': model,
                        'boundary': boundary,
                        'source_type': source_type,
                        'target_type': target_type,
                        'source_success_rate': source_success * 100,
                        'target_success_rate': target_success * 100,
                        'relative_effectiveness': relative_effectiveness * 100,
                        'protection_consistency': protection_consistency * 100
                    })
    
    return pd.DataFrame(detailed_metrics)

def calculate_boundary_effectiveness_comparison(results_df):
    """
    Calculate relative effectiveness between different boundary types.
    
    Args:
        results_df: DataFrame containing experiment results
    
    Returns:
        DataFrame with boundary effectiveness comparison metrics
    """
    comparison_metrics = []
    
    # Get unique boundary types
    boundary_types = results_df['boundary'].unique()
    boundary_types = boundary_types[boundary_types != 'none']  # Exclude no-boundary case
    
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        
        for attack_type in model_df['attack_type'].unique():
            type_df = model_df[model_df['attack_type'] == attack_type]
            
            # Calculate effectiveness for each boundary type
            boundary_effectiveness = {}
            for boundary in boundary_types:
                boundary_df = type_df[type_df['boundary'] == boundary]
                if len(boundary_df) > 0:
                    boundary_effectiveness[boundary] = 1 - boundary_df['attack_success'].mean()
            
            # Calculate relative effectiveness between boundary types
            for b1 in boundary_types:
                for b2 in boundary_types:
                    if b1 >= b2 or b1 not in boundary_effectiveness or b2 not in boundary_effectiveness:
                        continue
                    
                    eff1 = boundary_effectiveness[b1]
                    eff2 = boundary_effectiveness[b2]
                    
                    relative_effectiveness = (eff1 - eff2) / eff1 if eff1 > 0 else 0
                    
                    comparison_metrics.append({
                        'model': model,
                        'attack_type': attack_type,
                        'boundary1': b1,
                        'boundary2': b2,
                        'effectiveness1': eff1 * 100,
                        'effectiveness2': eff2 * 100,
                        'relative_effectiveness': relative_effectiveness * 100
                    })
    
    return pd.DataFrame(comparison_metrics)

def generate_cross_modal_heatmap(metrics_df, output_path):
    """
    Generate a heatmap visualization of cross-modal transfer effectiveness.
    
    Args:
        metrics_df: DataFrame containing cross-modal metrics
        output_path: Path to save the output visualization
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create pivot table for heatmap
    pivot_data = metrics_df.pivot_table(
        values='relative_effectiveness',
        index='source_type',
        columns='target_type',
        aggfunc='mean'
    )
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        pivot_data,
        annot=True,
        cmap='RdYlGn_r',
        center=0,
        fmt='.1f',
        cbar_kws={'label': 'Relative Effectiveness (%)'}
    )
    
    plt.title('Cross-Modal Transfer Effectiveness')
    plt.xlabel('Target Type')
    plt.ylabel('Source Type')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_boundary_comparison_chart(metrics_df, output_path):
    """
    Generate a comparison chart of boundary effectiveness.
    
    Args:
        metrics_df: DataFrame containing boundary comparison metrics
        output_path: Path to save the output visualization
    """
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Create grouped bar chart
    plt.figure(figsize=(12, 6))
    
    sns.barplot(
        data=metrics_df,
        x='attack_type',
        y='relative_effectiveness',
        hue='boundary1',
        palette='muted'
    )
    
    plt.title('Relative Boundary Effectiveness by Attack Type')
    plt.xlabel('Attack Type')
    plt.ylabel('Relative Effectiveness (%)')
    plt.xticks(rotation=45)
    plt.legend(title='Boundary Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save visualization
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def detailed_cross_modal_metrics(results_df):
    """
    Calculate transfer effectiveness and protection degradation between each modality pair.
    Args:
        results_df: DataFrame with experiment results (must include 'attack_type', 'boundary', 'model', 'attack_success')
    Returns:
        DataFrame with source_modality, target_modality, boundary, model, transfer_effectiveness, protection_degradation, sample_count
    """
    metrics = []
    modalities = results_df['attack_type'].unique()
    for model in results_df['model'].unique():
        for boundary in results_df['boundary'].unique():
            for src in modalities:
                for tgt in modalities:
                    if src == tgt:
                        continue
                    src_df = results_df[(results_df['attack_type'] == src) & (results_df['boundary'] == boundary) & (results_df['model'] == model)]
                    tgt_df = results_df[(results_df['attack_type'] == tgt) & (results_df['boundary'] == boundary) & (results_df['model'] == model)]
                    if len(src_df) == 0 or len(tgt_df) == 0:
                        continue
                    src_rate = src_df['attack_success'].mean()
                    tgt_rate = tgt_df['attack_success'].mean()
                    transfer_effectiveness = 1 - (tgt_rate / src_rate) if src_rate > 0 else np.nan
                    protection_degradation = tgt_rate - src_rate
                    metrics.append({
                        'model': model,
                        'boundary': boundary,
                        'source_modality': src,
                        'target_modality': tgt,
                        'transfer_effectiveness': transfer_effectiveness,
                        'protection_degradation': protection_degradation,
                        'sample_count': min(len(src_df), len(tgt_df))
                    })
    return pd.DataFrame(metrics)

# --- Visualization Functions ---
def create_transfer_heatmap(metrics_df, output_path):
    """
    Generate a heatmap showing transfer effectiveness between modalities.
    Args:
        metrics_df: DataFrame from detailed_cross_modal_metrics
        output_path: Path to save the heatmap
    """
    if metrics_df.empty:
        print("[WARN] No data for transfer heatmap.")
        return
    pivot = metrics_df.pivot_table(
        values='transfer_effectiveness',
        index='source_modality',
        columns='target_modality',
        aggfunc='mean'
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, cmap='coolwarm', fmt='.2f', cbar_kws={'label': 'Transfer Effectiveness'})
    plt.title('Cross-Modal Transfer Effectiveness')
    plt.xlabel('Target Modality')
    plt.ylabel('Source Modality')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_boundary_comparison_chart(metrics_df, output_path):
    """
    Compare different boundaries across modalities using a grouped bar chart.
    Args:
        metrics_df: DataFrame from detailed_cross_modal_metrics
        output_path: Path to save the chart
    """
    if metrics_df.empty:
        print("[WARN] No data for boundary comparison chart.")
        return
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=metrics_df,
        x='target_modality',
        y='transfer_effectiveness',
        hue='boundary',
        ci=None
    )
    plt.title('Boundary Comparison Across Modalities')
    plt.xlabel('Target Modality')
    plt.ylabel('Transfer Effectiveness')
    plt.legend(title='Boundary')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def create_protection_radar(metrics_df, output_path, boundary=None, model=None):
    """
    Generate radar chart of protection by modality for a given boundary/model.
    Args:
        metrics_df: DataFrame from detailed_cross_modal_metrics
        output_path: Path to save the radar chart
        boundary: Optional, filter by boundary
        model: Optional, filter by model
    """
    df = metrics_df.copy()
    if boundary:
        df = df[df['boundary'] == boundary]
    if model:
        df = df[df['model'] == model]
    if df.empty:
        print("[WARN] No data for radar chart.")
        return
    modalities = sorted(set(df['target_modality']))
    values = [1 - df[df['target_modality'] == m]['protection_degradation'].mean() for m in modalities]
    values += values[:1]  # close the loop
    angles = np.linspace(0, 2 * np.pi, len(modalities) + 1)
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, 'o-', linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180/np.pi, modalities)
    plt.title(f'Protection Radar{f" ({boundary})" if boundary else ""}{f" [{model}]" if model else ""}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

# --- Statistical Analysis Functions ---
def calculate_significance(results_df, group_col='boundary', value_col='attack_success'):
    """
    Determine if differences between boundaries are statistically significant using t-tests.
    Args:
        results_df: DataFrame with experiment results
        group_col: Column to group by (default: 'boundary')
        value_col: Column to test (default: 'attack_success')
    Returns:
        DataFrame with pairs, p-values, and significance
    """
    groups = results_df[group_col].unique()
    results = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            vals1 = results_df[results_df[group_col] == g1][value_col].dropna()
            vals2 = results_df[results_df[group_col] == g2][value_col].dropna()
            if len(vals1) < 2 or len(vals2) < 2:
                continue
            stat, pval = ttest_ind(vals1, vals2, equal_var=False)
            results.append({
                'group1': g1,
                'group2': g2,
                'p_value': pval,
                'significant': pval < 0.05
            })
    return pd.DataFrame(results)

def find_vulnerability_patterns(results_df):
    """
    Identify patterns in successful attacks (e.g., by attack type, boundary, model).
    Args:
        results_df: DataFrame with experiment results
    Returns:
        DataFrame with pattern, count, and success rate
    """
    patterns = results_df[results_df['attack_success'] == 1]
    summary = patterns.groupby(['attack_type', 'boundary', 'model']).size().reset_index(name='count')
    total = results_df.groupby(['attack_type', 'boundary', 'model']).size().reset_index(name='total')
    merged = pd.merge(summary, total, on=['attack_type', 'boundary', 'model'], how='left')
    merged['success_rate'] = merged['count'] / merged['total']
    return merged.sort_values('success_rate', ascending=False)

def boundary_effectiveness_score(results_df):
    """
    Create a composite score for each boundary type (lower is better).
    Args:
        results_df: DataFrame with experiment results
    Returns:
        DataFrame with boundary, score
    """
    grouped = results_df.groupby('boundary')['attack_success'].mean().reset_index()
    grouped['effectiveness_score'] = 1 - grouped['attack_success']
    return grouped[['boundary', 'effectiveness_score']]

# --- Model Comparison Metrics ---
def model_vulnerability_profile(results_df):
    """
    Generate vulnerability profiles for each model.
    Args:
        results_df: DataFrame with experiment results
    Returns:
        DataFrame with model, attack_type, boundary, vulnerability_rate
    """
    profile = results_df.groupby(['model', 'attack_type', 'boundary'])['attack_success'].mean().reset_index()
    profile.rename(columns={'attack_success': 'vulnerability_rate'}, inplace=True)
    return profile

def differential_analysis(results_df):
    """
    Compare how models differ in vulnerability for each attack type and boundary.
    Args:
        results_df: DataFrame with experiment results
    Returns:
        DataFrame with model1, model2, attack_type, boundary, diff
    """
    models = results_df['model'].unique()
    diffs = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            for attack_type in results_df['attack_type'].unique():
                for boundary in results_df['boundary'].unique():
                    v1 = results_df[(results_df['model'] == m1) & (results_df['attack_type'] == attack_type) & (results_df['boundary'] == boundary)]['attack_success'].mean()
                    v2 = results_df[(results_df['model'] == m2) & (results_df['attack_type'] == attack_type) & (results_df['boundary'] == boundary)]['attack_success'].mean()
                    if np.isnan(v1) or np.isnan(v2):
                        continue
                    diffs.append({
                        'model1': m1,
                        'model2': m2,
                        'attack_type': attack_type,
                        'boundary': boundary,
                        'vulnerability_diff': v1 - v2
                    })
    return pd.DataFrame(diffs)

def create_model_comparison_visualization(diffs_df, output_path):
    """
    Visualize model differences in vulnerability as a heatmap.
    Args:
        diffs_df: DataFrame from differential_analysis
        output_path: Path to save the visualization
    """
    if diffs_df.empty:
        print("[WARN] No data for model comparison visualization.")
        return
    pivot = diffs_df.pivot_table(
        values='vulnerability_diff',
        index=['model1', 'model2'],
        columns='attack_type',
        aggfunc='mean'
    )
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, cmap='bwr', center=0, fmt='.2f', cbar_kws={'label': 'Vulnerability Difference'})
    plt.title('Model Vulnerability Differential Analysis')
    plt.xlabel('Attack Type')
    plt.ylabel('Model Pair')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def boundary_effectiveness_by_attack_category(results_df):
    """
    Calculate effectiveness score for each boundary type against each attack category and subtype.
    Args:
        results_df: DataFrame with experiment results (must include 'boundary', 'attack_type', 'attack_subtype', 'attack_success')
    Returns:
        DataFrame with boundary, attack_type, attack_subtype, effectiveness_score, sample_count
    """
    grouped = results_df.groupby(['boundary', 'attack_type', 'attack_subtype'])['attack_success']
    summary = grouped.agg(['mean', 'count']).reset_index()
    summary.rename(columns={'mean': 'attack_success_rate', 'count': 'sample_count'}, inplace=True)
    summary['effectiveness_score'] = 1 - summary['attack_success_rate']
    return summary[['boundary', 'attack_type', 'attack_subtype', 'effectiveness_score', 'sample_count']]

def compare_boundary_effectiveness_across_categories(results_df):
    """
    Create a pivot table comparing boundary effectiveness across attack categories and subtypes.
    Args:
        results_df: DataFrame with experiment results
    Returns:
        Pivot table (DataFrame) with boundaries as rows, attack_type/attack_subtype as columns, values as effectiveness_score
    """
    eff_df = boundary_effectiveness_by_attack_category(results_df)
    pivot = eff_df.pivot_table(
        values='effectiveness_score',
        index='boundary',
        columns=['attack_type', 'attack_subtype'],
        aggfunc='mean'
    )
    return pivot

def cross_modal_boundary_transfer_analysis(results_df):
    """
    Analyze how well boundary effectiveness transfers between attack types (modalities).
    For each boundary, computes the transfer effectiveness from each source attack type to each target attack type.
    Transfer effectiveness is defined as 1 - (target_attack_success_rate / source_attack_success_rate),
    expressed as a percentage (higher = better transfer of protection).
    Args:
        results_df: DataFrame with experiment results (must include 'boundary', 'attack_type', 'attack_success')
    Returns:
        DataFrame with boundary, source_attack_type, target_attack_type, transfer_effectiveness (percent), sample_count
    """
    boundaries = results_df['boundary'].unique()
    attack_types = results_df['attack_type'].unique()
    rows = []
    for boundary in boundaries:
        bdf = results_df[results_df['boundary'] == boundary]
        for src in attack_types:
            src_df = bdf[bdf['attack_type'] == src]
            src_rate = src_df['attack_success'].mean()
            for tgt in attack_types:
                if src == tgt:
                    continue
                tgt_df = bdf[bdf['attack_type'] == tgt]
                tgt_rate = tgt_df['attack_success'].mean()
                if src_df.empty or tgt_df.empty or src_rate == 0:
                    continue
                transfer_effectiveness = 1 - (tgt_rate / src_rate)
                rows.append({
                    'boundary': boundary,
                    'source_attack_type': src,
                    'target_attack_type': tgt,
                    'transfer_effectiveness': transfer_effectiveness * 100,
                    'sample_count': min(len(src_df), len(tgt_df))
                })
    return pd.DataFrame(rows)

def cross_modal_transfer_matrix(results_df):
    """
    Create a pivot table (matrix) of cross-modal transfer effectiveness for each boundary.
    Args:
        results_df: DataFrame with experiment results
    Returns:
        Dict of DataFrames: {boundary: matrix DataFrame (rows=source_attack_type, cols=target_attack_type, values=transfer_effectiveness)}
    """
    df = cross_modal_boundary_transfer_analysis(results_df)
    matrices = {}
    for boundary in df['boundary'].unique():
        bdf = df[df['boundary'] == boundary]
        matrix = bdf.pivot_table(
            values='transfer_effectiveness',
            index='source_attack_type',
            columns='target_attack_type',
            aggfunc='mean'
        )
        matrices[boundary] = matrix
    return matrices

def attack_pattern_vulnerability_analysis(results_df, top_n=10):
    """
    Analyze and summarize attack patterns:
    - Groups successful attacks by attack_type, model, and boundary
    - Identifies which combinations are most vulnerable
    - Generates summary statistics (count, success rate, rank)
    Args:
        results_df: DataFrame with experiment results (must include 'attack_success', 'attack_type', 'model', 'boundary')
        top_n: Number of top vulnerable patterns to return (default 10)
    Returns:
        DataFrame with columns: attack_type, model, boundary, success_count, total_count, success_rate, rank
    """
    grouped = results_df.groupby(['attack_type', 'model', 'boundary'])['attack_success']
    summary = grouped.agg(['sum', 'count', 'mean']).reset_index()
    summary.rename(columns={'sum': 'success_count', 'count': 'total_count', 'mean': 'success_rate'}, inplace=True)
    summary['success_rate'] = summary['success_rate'] * 100
    summary = summary.sort_values(['success_rate', 'success_count'], ascending=[False, False]).reset_index(drop=True)
    summary['rank'] = summary.index + 1
    if top_n is not None:
        summary = summary.head(top_n)
    return summary[['rank', 'attack_type', 'model', 'boundary', 'success_count', 'total_count', 'success_rate']]

def boundary_significance_tests(results_df, group_col='boundary', value_col='attack_success'):
    """
    Perform pairwise t-tests between all boundary types to determine if differences are statistically significant.
    Args:
        results_df: DataFrame with experiment results
        group_col: Column to group by (default: 'boundary')
        value_col: Column to test (default: 'attack_success')
    Returns:
        DataFrame with boundary1, boundary2, p_value, significant (p<0.05)
    """
    from scipy.stats import ttest_ind
    groups = results_df[group_col].unique()
    results = []
    for i, g1 in enumerate(groups):
        for g2 in groups[i+1:]:
            vals1 = results_df[results_df[group_col] == g1][value_col].dropna()
            vals2 = results_df[results_df[group_col] == g2][value_col].dropna()
            if len(vals1) < 2 or len(vals2) < 2:
                continue
            stat, pval = ttest_ind(vals1, vals2, equal_var=False)
            results.append({
                'boundary1': g1,
                'boundary2': g2,
                'p_value': pval,
                'significant': pval < 0.05
            })
    return pd.DataFrame(results)

def boundary_confidence_intervals(results_df, group_cols=['boundary'], value_col='attack_success', confidence=0.95):
    """
    Calculate confidence intervals for attack success rates for each boundary (optionally by attack_type/model).
    Args:
        results_df: DataFrame with experiment results
        group_cols: List of columns to group by (default: ['boundary'])
        value_col: Column to calculate CI for (default: 'attack_success')
        confidence: Confidence level (default: 0.95)
    Returns:
        DataFrame with group columns, mean, lower_ci, upper_ci, sample_count
    """
    from scipy.stats import norm
    grouped = results_df.groupby(group_cols)[value_col]
    summary = grouped.agg(['mean', 'count', 'std']).reset_index()
    z = norm.ppf(1 - (1-confidence)/2)
    summary['stderr'] = summary['std'] / np.sqrt(summary['count'])
    summary['lower_ci'] = summary['mean'] - z * summary['stderr']
    summary['upper_ci'] = summary['mean'] + z * summary['stderr']
    summary.rename(columns={'mean': 'success_rate', 'count': 'sample_count'}, inplace=True)
    return summary[group_cols + ['success_rate', 'lower_ci', 'upper_ci', 'sample_count']]

def model_attack_response_matrix(results_df):
    """
    Create a matrix of attack success rates for each model, attack_type, and boundary.
    Args:
        results_df: DataFrame with experiment results
    Returns:
        DataFrame with models as rows, (attack_type, boundary) as columns, values as attack success rate
    """
    matrix = results_df.pivot_table(
        values='attack_success',
        index='model',
        columns=['attack_type', 'boundary'],
        aggfunc='mean'
    )
    return matrix

def model_specific_vulnerabilities(results_df, top_n=10):
    """
    Identify model-specific vulnerabilities: for each model, find the attack_type/boundary combinations with the highest attack success rates.
    Args:
        results_df: DataFrame with experiment results
        top_n: Number of most vulnerable combinations to return per model (default 10)
    Returns:
        DataFrame with model, attack_type, boundary, success_rate, sample_count, rank (per model)
    """
    grouped = results_df.groupby(['model', 'attack_type', 'boundary'])['attack_success']
    summary = grouped.agg(['mean', 'count']).reset_index()
    summary.rename(columns={'mean': 'success_rate', 'count': 'sample_count'}, inplace=True)
    summary['success_rate'] = summary['success_rate'] * 100
    summary = summary.sort_values(['model', 'success_rate', 'sample_count'], ascending=[True, False, False])
    summary['rank'] = summary.groupby('model').cumcount() + 1
    if top_n is not None:
        summary = summary[summary['rank'] <= top_n]
    return summary[['model', 'attack_type', 'boundary', 'success_rate', 'sample_count', 'rank']]

def model_differential_analysis(results_df):
    """
    For each model pair, attack_type, and boundary, compute the difference in attack success rates (vulnerability_diff).
    Args:
        results_df: DataFrame with experiment results
    Returns:
        DataFrame with model1, model2, attack_type, boundary, vulnerability_diff
    """
    models = results_df['model'].unique()
    diffs = []
    for i, m1 in enumerate(models):
        for m2 in models[i+1:]:
            for attack_type in results_df['attack_type'].unique():
                for boundary in results_df['boundary'].unique():
                    v1 = results_df[(results_df['model'] == m1) & (results_df['attack_type'] == attack_type) & (results_df['boundary'] == boundary)]['attack_success'].mean()
                    v2 = results_df[(results_df['model'] == m2) & (results_df['attack_type'] == attack_type) & (results_df['boundary'] == boundary)]['attack_success'].mean()
                    if np.isnan(v1) or np.isnan(v2):
                        continue
                    diffs.append({
                        'model1': m1,
                        'model2': m2,
                        'attack_type': attack_type,
                        'boundary': boundary,
                        'vulnerability_diff': v1 - v2
                    })
    return pd.DataFrame(diffs)

def export_results_with_metadata(results, config, output_dir, run_metadata=None):
    """
    Export experiment results and all relevant metadata to CSV and JSON for external/statistical analysis.
    Args:
        results: list of dicts or DataFrame of experiment results
        config: experiment config dict
        output_dir: directory to write export files
        run_metadata: optional dict of run metadata (e.g., git commit, timestamp)
    Outputs:
        - results.csv: All experiment results (flat table)
        - results.json: All experiment results (list of dicts)
        - experiment_metadata.json: Experiment config and run metadata
    """
    import os, json
    import pandas as pd
    os.makedirs(output_dir, exist_ok=True)
    # Convert to DataFrame if needed
    if not isinstance(results, pd.DataFrame):
        results_df = pd.DataFrame(results)
    else:
        results_df = results
    # Export results
    results_df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
    results_df.to_json(os.path.join(output_dir, 'results.json'), orient='records', indent=2)
    # Export metadata
    metadata = {
        'config': config,
        'run_metadata': run_metadata or {}
    }
    with open(os.path.join(output_dir, 'experiment_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

def calculate_implementation_complexity_score(boundary_type: str, prompt_length: int, processing_time: float) -> float:
    """
    Calculate a normalized implementation complexity score for a boundary.
    The score combines prompt length and processing time (min-max scaled and averaged).
    Args:
        boundary_type: The type of boundary (for future extensibility)
        prompt_length: Length of the prompt (int)
        processing_time: Time taken to process/apply the boundary (float, seconds)
    Returns:
        Normalized complexity score (float, 0-1)
    """
    # Example normalization (min-max scaling with arbitrary min/max for demo)
    min_length, max_length = 50, 2000
    min_time, max_time = 0.01, 2.0
    norm_length = (prompt_length - min_length) / (max_length - min_length)
    norm_time = (processing_time - min_time) / (max_time - min_time)
    norm_length = min(max(norm_length, 0), 1)
    norm_time = min(max(norm_time, 0), 1)
    score = 0.5 * norm_length + 0.5 * norm_time
    return score

def calculate_statistical_significance(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform chi-squared and Fisher's exact tests on attack success rates by boundary and attack_type.
    Args:
        results_df: DataFrame with columns ['boundary', 'attack_type', 'attack_success']
    Returns:
        DataFrame with boundary, attack_type, p_chi2, p_fisher, test_used
    """
    results = []
    for (boundary, attack_type), group in results_df.groupby(['boundary', 'attack_type']):
        success = group['attack_success'].sum()
        fail = len(group) - success
        # Compare to overall success/fail for other boundaries/attacks
        other = results_df[(results_df['boundary'] != boundary) | (results_df['attack_type'] != attack_type)]
        other_success = other['attack_success'].sum()
        other_fail = len(other) - other_success
        table = [[success, fail], [other_success, other_fail]]
        try:
            chi2, p_chi2, _, _ = chi2_contingency(table)
        except Exception:
            p_chi2 = None
        try:
            _, p_fisher = fisher_exact(table)
        except Exception:
            p_fisher = None
        test_used = 'fisher' if min(success, fail, other_success, other_fail) < 5 else 'chi2'
        results.append({
            'boundary': boundary,
            'attack_type': attack_type,
            'p_chi2': p_chi2,
            'p_fisher': p_fisher,
            'test_used': test_used
        })
    return pd.DataFrame(results)

def calculate_confidence_intervals(success_rates: List[float], confidence_level: float) -> List[Tuple[float, float]]:
    """
    Calculate confidence intervals for a list of attack success rates (proportions).
    Args:
        success_rates: List of success rates (floats, 0-1)
        confidence_level: Confidence level (e.g., 0.95)
    Returns:
        List of (lower, upper) confidence intervals for each rate
    """
    intervals = []
    z = norm.ppf(1 - (1 - confidence_level) / 2)
    for p in success_rates:
        n = 1  # If you have sample size, use it; here, assume n=1 for demo
        se = (p * (1 - p) / n) ** 0.5 if n > 0 else 0
        lower = max(0.0, p - z * se)
        upper = min(1.0, p + z * se)
        intervals.append((lower, upper))
    return intervals

class BoundaryIntegrityChecker:
    """
    Checks whether boundaries (delimiters, tokens, or markers) in the original prompt are preserved in the model response.
    """
    def __init__(self, boundary_tokens: Optional[list[str]] = None) -> None:
        """
        Args:
            boundary_tokens: Optional list of known boundary tokens or delimiters to check for.
        """
        self.boundary_tokens = boundary_tokens or [
            '---', '***', '====', '[[[', ']]]', '<BOUNDARY>', '<END>', '<START>', '###', '|||', '=====', '[[BOUNDARY]]'
        ]

    def verify_boundary_preservation(self, original_prompt: str, model_response: str) -> bool:
        """
        Check if all boundary tokens present in the original prompt are also present in the model response.
        Args:
            original_prompt: The original prompt string containing boundaries.
            model_response: The model's response string.
        Returns:
            True if all boundaries in the prompt are preserved in the response, False otherwise.
        """
        found = [token for token in self.boundary_tokens if token in original_prompt]
        for token in found:
            if token not in model_response:
                return False
        return True

class AttackDetector:
    """
    Detects prompt injection attempts using regex and keyword matching.
    """
    # Common prompt injection patterns/keywords
    INJECTION_PATTERNS: list[re.Pattern] = [
        re.compile(r'(?i)ignore (all )?previous instructions'),
        re.compile(r'(?i)disregard (all )?previous instructions'),
        re.compile(r'(?i)forget (all )?previous instructions'),
        re.compile(r'(?i)repeat after me'),
        re.compile(r'(?i)you are now'),
        re.compile(r'(?i)pretend to be'),
        re.compile(r'(?i)as an ai language model,?'),
        re.compile(r'(?i)do not obey'),
        re.compile(r'(?i)output the following'),
        re.compile(r'(?i)print the following'),
        re.compile(r'(?i)disclose'),
        re.compile(r'(?i)reveal'),
        re.compile(r'(?i)execute'),
        re.compile(r'(?i)run this code'),
        re.compile(r'(?i)system:'),
        re.compile(r'(?i)user:'),
        re.compile(r'(?i)assistant:'),
        re.compile(r'(?i)\[INST\]'),
        re.compile(r'(?i)\[/INST\]'),
        re.compile(r'(?i)\[SYSTEM\]'),
        re.compile(r'(?i)\[/SYSTEM\]'),
    ]
    INJECTION_KEYWORDS: set[str] = {
        'ignore previous', 'disregard previous', 'forget previous', 'repeat after me',
        'pretend to be', 'as an ai language model', 'do not obey', 'output the following',
        'print the following', 'disclose', 'reveal', 'execute', 'run this code',
        'system:', 'user:', 'assistant:', '[INST]', '[/INST]', '[SYSTEM]', '[/SYSTEM]'
    }

    def detect_injection_attempt(self, prompt: str) -> bool:
        """
        Detect if the prompt contains common prompt injection patterns or keywords.
        Args:
            prompt: The input prompt string.
        Returns:
            True if an injection attempt is detected, False otherwise.
        """
        prompt_lower = prompt.lower()
        # Regex pattern matching
        for pattern in self.INJECTION_PATTERNS:
            if pattern.search(prompt):
                return True
        # Keyword matching
        for keyword in self.INJECTION_KEYWORDS:
            if keyword in prompt_lower:
                return True
        return False