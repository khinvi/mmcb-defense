import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

def generate_summary_chart(results_df, output_path):
    """
    Generate a summary chart showing attack success rates by model and boundary type.
    Only for structured data and code attacks.
    """
    # Prepare data
    chart_data = []
    # Calculate success rate for each combination of model, boundary, and attack type
    for model in results_df['model'].unique():
        model_df = results_df[results_df['model'] == model]
        for boundary in model_df['boundary'].unique():
            boundary_df = model_df[model_df['boundary'] == boundary]
            # Skip rows with errors
            valid_df = boundary_df[boundary_df['attack_success'].notna()]
            if len(valid_df) == 0:
                continue
            for attack_type in valid_df['attack_type'].unique():
                if attack_type not in ['json', 'csv', 'yaml', 'xml', 'python', 'javascript']:
                    continue
                type_df = valid_df[valid_df['attack_type'] == attack_type]
                success_rate = type_df['attack_success'].mean() * 100
                chart_data.append({
                    'Model': model,
                    'Boundary': boundary,
                    'Attack Type': attack_type,
                    'Success Rate (%)': success_rate
                })
    chart_df = pd.DataFrame(chart_data)
    # Create the chart
    plt.figure(figsize=(12, 8))
    # Create grouped bar chart
    ax = sns.catplot(
        data=chart_df,
        kind="bar",
        x="Model",
        y="Success Rate (%)",
        hue="Boundary",
        col="Attack Type",
        height=6,
        aspect=0.8,
        palette="muted",
        legend=False
    )
    # Set chart title and labels
    ax.fig.suptitle('Attack Success Rate by Model, Boundary, and Attack Type', fontsize=16)
    ax.set_xlabels('Model', fontsize=12)
    ax.set_ylabels('Success Rate (%)', fontsize=12)
    # Add legend with descriptive titles
    plt.legend(
        title='Boundary Type',
        loc='upper right',
        fontsize=10
    )
    # Add descriptive text
    plt.figtext(
        0.5, 0.01,
        "Lower success rate indicates better boundary protection",
        ha='center',
        fontsize=12
    )
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_cross_modal_transfer_chart(metrics_df, output_path):
    """
    Generate a chart comparing the effectiveness of boundary transfer across structured data and code modalities only.
    """
    # Filter for transfer effectiveness metrics for structured data and code only
    transfer_df = metrics_df[(metrics_df['metric_type'] == 'transfer_effectiveness') &
                             (metrics_df['attack_type'].isin(['json', 'csv', 'yaml', 'xml', 'python', 'javascript']))]
    if len(transfer_df) == 0:
        return
    # Create the chart
    plt.figure(figsize=(10, 6))
    # Create grouped bar chart
    ax = sns.barplot(
        data=transfer_df,
        x="boundary",
        y="value",
        hue="attack_type",
        palette="muted"
    )
    # Set chart title and labels
    plt.title('Boundary Transfer Effectiveness by Modality (Structured Data & Code)', fontsize=16)
    plt.xlabel('Boundary Type', fontsize=12)
    plt.ylabel('Transfer Effectiveness (%)', fontsize=12)
    # Add legend with descriptive titles
    plt.legend(
        title='Attack Type',
        loc='upper right',
        fontsize=10
    )
    # Add descriptive text
    plt.figtext(
        0.5, 0.01,
        "Higher transfer effectiveness indicates better cross-modal boundary protection",
        ha='center',
        fontsize=10
    )
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_implementation_complexity_chart(metrics_df, output_path):
    """
    Generate a chart showing implementation complexity metrics for different boundary types.
    Only for structured data and code attacks.
    """
    # Only keep relevant boundaries and attack types
    metrics_df = metrics_df[metrics_df['attack_type'].isin(['json', 'csv', 'yaml', 'xml', 'python', 'javascript'])]
    # Create the chart
    plt.figure(figsize=(8, 5))
    # Create bar chart
    ax = sns.barplot(
        data=metrics_df,
        x="boundary",
        y="value",
        palette="muted"
    )
    # Set chart title and labels
    plt.title('Implementation Complexity by Boundary Type', fontsize=16)
    plt.xlabel('Boundary Type', fontsize=12)
    plt.ylabel('Average Prompt Length (characters)', fontsize=12)
    # Add descriptive text
    plt.figtext(
        0.5, 0.01,
        "Higher prompt length indicates higher implementation complexity",
        ha='center',
        fontsize=10
    )
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_attack_success_heatmap(results_df, output_path):
    """
    Generate a heatmap of attack success rates by boundary and attack type.
    Rows: boundary, Columns: attack_type, Values: mean attack success rate (%).
    Publication-quality for academic papers.
    """
    pivot = results_df.pivot_table(
        values='attack_success',
        index='boundary',
        columns='attack_type',
        aggfunc='mean'
    ) * 100
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'Attack Success Rate (%)'})
    plt.title('Attack Success Rate by Boundary and Attack Type', fontsize=18)
    plt.xlabel('Attack Type', fontsize=14)
    plt.ylabel('Boundary Type', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    plt.close()

def generate_cross_modal_effectiveness_matrix(transfer_df, output_dir):
    """
    Generate a heatmap matrix of cross-modal transfer effectiveness for each boundary.
    Rows: source_attack_type, Columns: target_attack_type, Values: transfer effectiveness (%).
    One heatmap per boundary. Publication-quality for academic papers.
    """
    boundaries = transfer_df['boundary'].unique()
    for boundary in boundaries:
        bdf = transfer_df[transfer_df['boundary'] == boundary]
        pivot = bdf.pivot_table(
            values='transfer_effectiveness',
            index='source_attack_type',
            columns='target_attack_type',
            aggfunc='mean'
        )
        plt.figure(figsize=(8, 6))
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Blues', cbar_kws={'label': 'Transfer Effectiveness (%)'})
        plt.title(f'Cross-Modal Transfer Effectiveness ({boundary})', fontsize=16)
        plt.xlabel('Target Attack Type', fontsize=13)
        plt.ylabel('Source Attack Type', fontsize=13)
        plt.xticks(fontsize=11)
        plt.yticks(fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'cross_modal_matrix_{boundary}.png'), dpi=400, bbox_inches='tight')
        plt.close()

def generate_model_vulnerability_profile(results_df, output_dir):
    """
    Generate radar/spider plots for each model showing vulnerability (attack success rate) across attack types and boundaries.
    Publication-quality for academic papers.
    """
    models = results_df['model'].unique()
    attack_types = results_df['attack_type'].unique()
    boundaries = results_df['boundary'].unique()
    for model in models:
        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
        for boundary in boundaries:
            rates = []
            for attack_type in attack_types:
                rate = results_df[(results_df['model'] == model) & (results_df['boundary'] == boundary) & (results_df['attack_type'] == attack_type)]['attack_success'].mean()
                rates.append(rate if not np.isnan(rate) else 0)
            # Close the loop
            rates += rates[:1]
            angles = np.linspace(0, 2 * np.pi, len(rates))
            ax.plot(angles, np.array(rates) * 100, label=boundary, linewidth=2)
            ax.fill(angles, np.array(rates) * 100, alpha=0.15)
        ax.set_xticks(np.linspace(0, 2 * np.pi, len(attack_types) + 1)[:-1])
        ax.set_xticklabels(list(attack_types) + [attack_types[0]], fontsize=12)
        ax.set_yticks(np.linspace(0, 100, 5))
        ax.set_yticklabels([f'{int(y)}%' for y in np.linspace(0, 100, 5)], fontsize=11)
        ax.set_title(f'Model Vulnerability Profile: {model}', fontsize=17, pad=20)
        ax.legend(title='Boundary', bbox_to_anchor=(1.1, 1.05), fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'model_vulnerability_{model}.png'), dpi=400, bbox_inches='tight')
        plt.close()

def generate_comprehensive_report(results_df, metrics, statistical_results, config, output_path):
    """
    Generate a comprehensive, publication-ready experiment report with:
    - Methodology (from config/metadata)
    - Results (key tables: attack success, cross-modal, vulnerabilities)
    - Statistical results (significance, confidence intervals)
    - Key insights (auto-summarized)
    All tables are formatted for research papers (Markdown).
    Args:
        results_df: DataFrame of experiment results
        metrics: dict of key metrics DataFrames (e.g., 'attack_success', 'cross_modal', 'vulnerabilities')
        statistical_results: dict of statistical results (e.g., 'significance', 'confidence_intervals')
        config: experiment config dict
        output_path: path to write the report (Markdown or .txt)
    """
    import datetime
    lines = []
    # --- Methodology ---
    lines.append('# Experiment Methodology\n')
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Models:** {', '.join(str(m) for m in config.get('models', []))}")
    lines.append(f"**Boundaries:** {', '.join(str(b) for b in config.get('boundaries', []))}")
    lines.append(f"**Attack Types:** {', '.join(str(a) for a in config.get('attack_types', []))}")
    lines.append(f"**Num Attacks per Combo:** {config.get('num_attacks', 'N/A')}")
    lines.append(f"**Batch Size:** {config.get('batch_size', 'N/A')}")
    lines.append(f"**Random Seed:** {config.get('seed', 'N/A')}")
    lines.append('\n---\n')
    # --- Results ---
    lines.append('# Results\n')
    # Attack Success Table
    if 'attack_success' in metrics:
        lines.append('## Attack Success Rates by Boundary and Attack Type\n')
        lines.append(metrics['attack_success'].to_markdown(index=True))
        lines.append('')
    # Cross-Modal Table
    if 'cross_modal' in metrics:
        lines.append('## Cross-Modal Transfer Effectiveness\n')
        for boundary, matrix in metrics['cross_modal'].items():
            lines.append(f'### Boundary: {boundary}')
            lines.append(matrix.to_markdown(index=True))
            lines.append('')
    # Model Vulnerabilities
    if 'vulnerabilities' in metrics:
        lines.append('## Top Model Vulnerabilities\n')
        lines.append(metrics['vulnerabilities'].to_markdown(index=False))
        lines.append('')
    # --- Statistical Results ---
    lines.append('# Statistical Analysis\n')
    if 'significance' in statistical_results:
        lines.append('## Boundary Significance Tests (p-values)\n')
        lines.append(statistical_results['significance'].to_markdown(index=False))
        lines.append('')
    if 'confidence_intervals' in statistical_results:
        lines.append('## Confidence Intervals for Success Rates\n')
        lines.append(statistical_results['confidence_intervals'].to_markdown(index=False))
        lines.append('')
    # --- Key Insights ---
    lines.append('# Key Insights\n')
    # Best and worst boundaries
    if 'attack_success' in metrics:
        attack_success = metrics['attack_success']
        best = attack_success.min().min()
        worst = attack_success.max().max()
        lines.append(f"- **Best boundary/attack combo:** {attack_success.stack().idxmin()} ({best:.1f}% success)")
        lines.append(f"- **Worst boundary/attack combo:** {attack_success.stack().idxmax()} ({worst:.1f}% success)")
    # Most vulnerable model
    if 'vulnerabilities' in metrics:
        top_vuln = metrics['vulnerabilities'].iloc[0]
        lines.append(f"- **Most vulnerable model:** {top_vuln['model']} (Attack: {top_vuln['attack_type']}, Boundary: {top_vuln['boundary']}, Success Rate: {top_vuln['success_rate']:.1f}%)")
    lines.append('\n---\n')
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))