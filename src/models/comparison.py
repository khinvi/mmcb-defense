"""
Comparison utilities for evaluating different models against the same attacks.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Tuple

class ModelComparison:
    """
    Utilities for comparing different models' performance against the same attacks.
    """
    
    def __init__(self, models: Dict[str, Any]):
        """
        Initialize the model comparison utility.
        
        Args:
            models: Dictionary mapping model names to model instances
        """
        self.models = models
        self.results = {}
    
    def compare_models(self, prompts: List[str], attack_instructions: List[str]) -> pd.DataFrame:
        """
        Compare multiple models against the same set of prompts and attack instructions.
        
        Args:
            prompts: List of prompts to test
            attack_instructions: Corresponding list of attack instructions
            
        Returns:
            DataFrame containing comparison results
        """
        results = []
        
        for i, (prompt, attack_instruction) in enumerate(zip(prompts, attack_instructions)):
            for model_name, model in self.models.items():
                try:
                    # Generate response from the model
                    response = model.generate_response(prompt)
                    
                    # Evaluate if attack was successful
                    attack_success = model.evaluate_attack_success(
                        prompt, attack_instruction, response
                    )
                    
                    # Record the result
                    result = {
                        'model': model_name,
                        'prompt_id': i,
                        'attack_success': attack_success,
                        'response_length': len(response),
                    }
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error comparing model {model_name} on prompt {i}: {str(e)}")
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        self.results = results_df
        
        return results_df
    
    def calculate_vulnerability_difference(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate the vulnerability differences between models.
        
        Returns:
            Tuple containing:
                - DataFrame with vulnerability differences by prompt
                - DataFrame with summary statistics
        """
        if len(self.results) == 0:
            raise ValueError("No comparison results available. Run compare_models first.")
            
        if len(self.models) < 2:
            raise ValueError("Need at least two models to calculate vulnerability differences.")
        
        # Pivot the results to get success rates by model and prompt
        pivot_df = pd.pivot_table(
            self.results,
            values='attack_success',
            index='prompt_id',
            columns='model',
            aggfunc=lambda x: int(any(x))  # Consider attack successful if any run succeeded
        )
        
        # Calculate the absolute difference between models for each prompt
        model_names = list(self.models.keys())
        diff_rows = []
        
        for prompt_id in pivot_df.index:
            row_values = pivot_df.loc[prompt_id].values
            
            # For each pair of models
            for i in range(len(model_names)):
                for j in range(i+1, len(model_names)):
                    model_i = model_names[i]
                    model_j = model_names[j]
                    
                    # Check if one model was vulnerable but not the other
                    if row_values[i] != row_values[j]:
                        diff_rows.append({
                            'prompt_id': prompt_id,
                            'model_1': model_i,
                            'model_2': model_j,
                            'model_1_vulnerable': bool(row_values[i]),
                            'model_2_vulnerable': bool(row_values[j]),
                            'differential_vulnerability': 1
                        })
                    else:
                        diff_rows.append({
                            'prompt_id': prompt_id,
                            'model_1': model_i,
                            'model_2': model_j,
                            'model_1_vulnerable': bool(row_values[i]),
                            'model_2_vulnerable': bool(row_values[j]),
                            'differential_vulnerability': 0
                        })
        
        # Convert to DataFrame
        diff_df = pd.DataFrame(diff_rows)
        
        # Calculate summary statistics
        summary_df = diff_df.groupby(['model_1', 'model_2'])['differential_vulnerability'].mean().reset_index()
        summary_df.columns = ['Model 1', 'Model 2', 'Differential Vulnerability Rate']
        
        return diff_df, summary_df
    
    def plot_comparison(self, output_path=None):
        """
        Plot the comparison of model vulnerabilities.
        
        Args:
            output_path: Optional path to save the plot
        """
        if len(self.results) == 0:
            raise ValueError("No comparison results available. Run compare_models first.")
        
        # Calculate success rates by model
        success_rates = self.results.groupby('model')['attack_success'].mean() * 100
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=success_rates.index, y=success_rates.values)
        
        # Customize the plot
        plt.title('Attack Success Rate by Model', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Attack Success Rate (%)', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for i, v in enumerate(success_rates.values):
            ax.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=12)
        
        # Add note that lower is better
        plt.figtext(0.5, 0.01, "Lower success rate indicates better protection", ha='center', fontsize=12)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_vulnerability_heatmap(self, attack_types, boundary_types, output_path=None):
        """
        Plot a heatmap of vulnerability differences between models across attack and boundary types.
        
        Args:
            attack_types: Dictionary mapping prompt_ids to attack types
            boundary_types: Dictionary mapping prompt_ids to boundary types
            output_path: Optional path to save the plot
        """
        if len(self.results) == 0:
            raise ValueError("No comparison results available. Run compare_models first.")
            
        # Add attack and boundary information to results
        enriched_results = self.results.copy()
        enriched_results['attack_type'] = enriched_results['prompt_id'].map(attack_types)
        enriched_results['boundary'] = enriched_results['prompt_id'].map(boundary_types)
        
        # Create pivot table
        heatmap_data = pd.pivot_table(
            enriched_results,
            values='attack_success',
            index=['attack_type', 'boundary'],
            columns=['model'],
            aggfunc=np.mean
        )
        
        # Plot the heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(heatmap_data, annot=True, fmt='.1%', cmap='YlOrRd', linewidths=.5)
        
        # Customize the plot
        plt.title('Attack Success Rate by Model, Attack Type, and Boundary', fontsize=16)
        plt.xlabel('Model', fontsize=14)
        plt.ylabel('Attack Type / Boundary', fontsize=14)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    def calculate_model_robustness_score(self) -> pd.DataFrame:
        """
        Calculate an overall robustness score for each model.
        
        Returns:
            DataFrame with robustness scores
        """
        if len(self.results) == 0:
            raise ValueError("No comparison results available. Run compare_models first.")
        
        # Calculate metrics for each model
        model_metrics = []
        
        for model_name in self.models.keys():
            model_df = self.results[self.results['model'] == model_name]
            
            # Success rate (lower is better)
            success_rate = model_df['attack_success'].mean()
            
            # Consistency (standard deviation of success rate)
            consistency = model_df['attack_success'].std()
            
            # Calculate robustness score (1 - success_rate) * (1 - consistency)
            # Higher score means more robust (fewer successful attacks and more consistent)
            robustness_score = (1 - success_rate) * (1 - consistency if consistency > 0 else 1)
            
            model_metrics.append({
                'model': model_name,
                'attack_success_rate': success_rate,
                'consistency': consistency,
                'robustness_score': robustness_score
            })
        
        return pd.DataFrame(model_metrics).sort_values('robustness_score', ascending=False)