import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def generate_summary_chart(results_df, output_path):
    """
    Generate a summary chart showing attack success rates by model and boundary type.
    
    Args:
        results_df: DataFrame containing experiment results
        output_path: Path to save the output chart
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
    Generate a chart comparing the effectiveness of boundary transfer across modalities.
    
    Args:
        metrics_df: DataFrame containing calculated metrics
        output_path: Path to save the output chart
    """
    # Filter for transfer effectiveness metrics
    transfer_df = metrics_df[metrics_df['metric_type'] == 'transfer_effectiveness']
    
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
    plt.title('Boundary Transfer Effectiveness by Modality', fontsize=16)
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
    
    Args:
        metrics_df: DataFrame containing complexity metrics
        output_path: Path to save the output chart
    """
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