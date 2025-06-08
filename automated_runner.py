#!/usr/bin/env python3
"""
Multi-Modal Context Boundaries (MMCB) Automated Experiment Runner
Runs experiments and notebooks 10 times, collecting structured summaries for analysis.
"""

import os
import sys
import subprocess
import json
import time
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging
import hashlib
import platform

class MMCBAutomatedRunner:
    def __init__(self, base_output_dir="automated_runs", num_runs=25, skip_notebooks=False):
        self.base_output_dir = Path(base_output_dir)
        self.num_runs = num_runs
        self.skip_notebooks = skip_notebooks
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Paths
        self.main_script = "src/main.py"
        self.notebook_path = "notebooks/analysis.ipynb"
        
    def setup_logging(self):
        """Setup logging for the automated runner."""
        log_file = self.base_output_dir / "automated_runner.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def get_system_info(self):
        """Collect system information for metadata."""
        return {
            "platform": platform.platform(),
            "python_version": sys.version,
            "cpu_count": os.cpu_count(),
            "timestamp": datetime.now().isoformat(),
            "user": os.getenv("USER", "unknown"),
            "hostname": platform.node()
        }
    
    def create_run_directory(self, run_number):
        """Create directory structure for a single run."""
        run_dir = self.base_output_dir / f"run_{run_number}"
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (run_dir / "data").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "notebooks").mkdir(exist_ok=True)
        (run_dir / "summaries").mkdir(exist_ok=True)
        
        return run_dir
    
    def run_experiment(self, run_number, run_dir):
        """Run the main MMCB experiment script."""
        self.logger.info(f"Starting experiment run {run_number}")
        
        start_time = time.time()
        
        # Prepare command with unique output directory and random seed
        cmd = [
            sys.executable, self.main_script,
            "--output_dir", str(run_dir / "data"),
            "--seed", str(42 + run_number),  # Reproducible but different seeds
            "--log", "info"
            # Remove --quick for more comprehensive experiments with M3 Pro power
        ]
        
        # Execute the experiment
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd="."
            )
            
            execution_time = time.time() - start_time
            
            # Save execution logs
            with open(run_dir / "logs" / "experiment_stdout.log", "w") as f:
                f.write(result.stdout)
            with open(run_dir / "logs" / "experiment_stderr.log", "w") as f:
                f.write(result.stderr)
            
            # Check if successful
            success = result.returncode == 0
            
            # Save execution metadata
            execution_metadata = {
                "run_number": run_number,
                "success": success,
                "return_code": result.returncode,
                "execution_time_seconds": execution_time,
                "command": " ".join(cmd),
                "timestamp": datetime.now().isoformat(),
                "system_info": self.get_system_info()
            }
            
            with open(run_dir / "execution_metadata.json", "w") as f:
                json.dump(execution_metadata, f, indent=2)
            
            if success:
                self.logger.info(f"Experiment run {run_number} completed successfully in {execution_time:.1f}s")
            else:
                self.logger.error(f"Experiment run {run_number} failed with return code {result.returncode}")
                
            return success, execution_metadata
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Experiment run {run_number} timed out after 1 hour")
            return False, {"error": "timeout", "run_number": run_number}
        except Exception as e:
            self.logger.error(f"Experiment run {run_number} failed with exception: {e}")
            return False, {"error": str(e), "run_number": run_number}
    
    def execute_notebook(self, run_number, run_dir):
        """Execute the analysis notebook and extract text summaries."""
        self.logger.info(f"Executing analysis notebook for run {run_number}")
        
        # Check if notebook exists
        if not os.path.exists(self.notebook_path):
            self.logger.error(f"Notebook not found at {self.notebook_path}")
            self.logger.info("Available notebooks:")
            for nb_file in Path(".").glob("**/*.ipynb"):
                self.logger.info(f"  Found: {nb_file}")
            return False
        
        start_time = time.time()
        
        # Create output notebook path
        output_notebook = run_dir / "notebooks" / "analysis_executed.ipynb"
        
        # Execute notebook using nbconvert
        cmd = [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=600",
            "--ExecutePreprocessor.allow_errors=True",  # Continue on errors
            "--output", str(output_notebook),
            self.notebook_path
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1200,  # 20 minute timeout
                cwd="."
            )
            
            execution_time = time.time() - start_time
            
            # Save notebook execution logs
            with open(run_dir / "logs" / "notebook_stdout.log", "w") as f:
                f.write(result.stdout)
            with open(run_dir / "logs" / "notebook_stderr.log", "w") as f:
                f.write(result.stderr)
            
            success = result.returncode == 0
            
            if success:
                self.logger.info(f"Notebook execution for run {run_number} completed in {execution_time:.1f}s")
                
                # Convert notebook to text and markdown for easy analysis
                self.convert_notebook_to_text(output_notebook, run_dir)
                
            else:
                self.logger.error(f"Notebook execution for run {run_number} failed with return code {result.returncode}")
                self.logger.error(f"STDOUT: {result.stdout}")
                self.logger.error(f"STDERR: {result.stderr}")
                
                # Still try to create summaries from experiment data even if notebook fails
                self.create_text_summary(run_dir)
                
            return success
            
        except subprocess.TimeoutExpired:
            self.logger.error(f"Notebook execution for run {run_number} timed out after 20 minutes")
            return False
        except FileNotFoundError as e:
            self.logger.error(f"Command not found: {e}")
            self.logger.error("Make sure 'jupyter' is installed and in your PATH")
            self.logger.error("Try: pip install jupyter nbconvert")
            return False
        except Exception as e:
            self.logger.error(f"Notebook execution for run {run_number} failed: {e}")
            return False
    
    def convert_notebook_to_text(self, notebook_path, run_dir):
        """Convert executed notebook to text formats for LLM analysis."""
        self.logger.info("Converting notebook to text formats")
        
        # Convert to markdown
        cmd_md = [
            "jupyter", "nbconvert",
            "--to", "markdown",
            "--output", str(run_dir / "summaries" / "analysis_summary.md"),
            str(notebook_path)
        ]
        
        # Convert to HTML (contains all outputs)
        cmd_html = [
            "jupyter", "nbconvert", 
            "--to", "html",
            "--output", str(run_dir / "summaries" / "analysis_full.html"),
            str(notebook_path)
        ]
        
        try:
            subprocess.run(cmd_md, check=True, capture_output=True)
            subprocess.run(cmd_html, check=True, capture_output=True)
            
            # Also create a plain text summary
            self.create_text_summary(run_dir)
            
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to convert notebook: {e}")
    
    def create_text_summary(self, run_dir):
        """Create a structured text summary of the run for LLM analysis."""
        
        # Load execution metadata
        metadata_file = run_dir / "execution_metadata.json"
        if metadata_file.exists():
            with open(metadata_file) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load any results data if available
        results_data = self.extract_results_data(run_dir)
        
        # Create structured summary
        summary = {
            "run_metadata": {
                "run_number": metadata.get("run_number", "unknown"),
                "execution_time_seconds": metadata.get("execution_time_seconds", 0),
                "success": metadata.get("success", False),
                "timestamp": metadata.get("timestamp", "unknown"),
                "system_info": metadata.get("system_info", {})
            },
            "experiment_results": results_data,
            "key_findings": self.extract_key_findings(run_dir),
            "performance_metrics": self.calculate_performance_metrics(results_data)
        }
        
        # Save as JSON for programmatic access
        with open(run_dir / "summaries" / "run_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save as formatted text for LLM analysis
        text_summary = self.format_text_summary(summary)
        with open(run_dir / "summaries" / "run_summary.txt", "w") as f:
            f.write(text_summary)
            
        self.logger.info("Created structured text summary")
    
    def extract_results_data(self, run_dir):
        """Extract results data from the experiment output."""
        results = {}
        
        # Look for results files in the data directory
        data_dir = run_dir / "data"
        if data_dir.exists():
            for results_file in data_dir.glob("**/results_*.json"):
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                        if isinstance(data, list) and len(data) > 0:
                            # Convert to DataFrame for analysis
                            df = pd.DataFrame(data)
                            results["raw_experiments"] = len(df)
                            results["attack_success_rate"] = df.get("attack_success", pd.Series()).mean()
                            results["boundary_types"] = df.get("boundary", pd.Series()).unique().tolist()
                            results["attack_types"] = df.get("attack_type", pd.Series()).unique().tolist()
                            results["models_tested"] = df.get("model", pd.Series()).unique().tolist()
                            break
                except Exception as e:
                    self.logger.warning(f"Could not load results file {results_file}: {e}")
        
        return results
    
    def extract_key_findings(self, run_dir):
        """Extract key findings from the analysis outputs."""
        findings = []
        
        # Look for markdown summary
        md_file = run_dir / "summaries" / "analysis_summary.md"
        if md_file.exists():
            try:
                with open(md_file) as f:
                    content = f.read()
                    
                # Extract sections that look like findings
                if "Key Findings" in content:
                    findings.append("Key findings section found in analysis")
                if "Recommendation" in content:
                    findings.append("Recommendations section found in analysis")
                if "Conclusion" in content:
                    findings.append("Conclusion section found in analysis")
                    
            except Exception as e:
                self.logger.warning(f"Could not extract findings from {md_file}: {e}")
        
        return findings
    
    def calculate_performance_metrics(self, results_data):
        """Calculate performance metrics from results."""
        metrics = {}
        
        if results_data:
            metrics["total_experiments"] = results_data.get("raw_experiments", 0)
            metrics["overall_attack_success_rate"] = results_data.get("attack_success_rate", 0)
            metrics["num_boundary_types"] = len(results_data.get("boundary_types", []))
            metrics["num_attack_types"] = len(results_data.get("attack_types", []))
            metrics["num_models"] = len(results_data.get("models_tested", []))
            
        return metrics
    
    def format_text_summary(self, summary):
        """Format summary as readable text for LLM analysis."""
        text = f"""
MMCB Experiment Run Summary
==========================

Run Metadata:
- Run Number: {summary['run_metadata']['run_number']}
- Execution Time: {summary['run_metadata']['execution_time_seconds']:.1f} seconds
- Success: {summary['run_metadata']['success']}
- Timestamp: {summary['run_metadata']['timestamp']}

Experiment Results:
- Total Experiments: {summary['experiment_results'].get('raw_experiments', 'N/A')}
- Overall Attack Success Rate: {summary['experiment_results'].get('attack_success_rate', 'N/A'):.3f}
- Boundary Types Tested: {', '.join(summary['experiment_results'].get('boundary_types', []))}
- Attack Types Tested: {', '.join(summary['experiment_results'].get('attack_types', []))}
- Models Tested: {', '.join(summary['experiment_results'].get('models_tested', []))}

Performance Metrics:
"""
        
        for key, value in summary['performance_metrics'].items():
            text += f"- {key.replace('_', ' ').title()}: {value}\n"
        
        text += f"""
Key Findings:
"""
        for finding in summary['key_findings']:
            text += f"- {finding}\n"
        
        text += f"""
System Information:
- Platform: {summary['run_metadata']['system_info'].get('platform', 'unknown')}
- Python Version: {summary['run_metadata']['system_info'].get('python_version', 'unknown')}
- CPU Count: {summary['run_metadata']['system_info'].get('cpu_count', 'unknown')}
"""
        
        return text
    
    def run_all_experiments(self):
        """Run all experiments and collect summaries."""
        self.logger.info(f"Starting automated run of {self.num_runs} experiments")
        
        successful_runs = 0
        failed_runs = []
        
        # Create overall metadata
        overall_metadata = {
            "start_time": datetime.now().isoformat(),
            "total_runs": self.num_runs,
            "system_info": self.get_system_info()
        }
        
        for run_num in range(1, self.num_runs + 1):
            self.logger.info(f"="*50)
            self.logger.info(f"Starting Run {run_num}/{self.num_runs}")
            self.logger.info(f"="*50)
            
            # Create run directory
            run_dir = self.create_run_directory(run_num)
            
            # Run experiment
            exp_success, exp_metadata = self.run_experiment(run_num, run_dir)
            
            if exp_success:
                # Run notebook analysis unless skipped
                if self.skip_notebooks:
                    self.logger.info(f"Skipping notebook execution for run {run_num} (skip_notebooks=True)")
                    # Still create summaries from experiment data
                    self.create_text_summary(run_dir)
                    notebook_success = True
                else:
                    notebook_success = self.execute_notebook(run_num, run_dir)
                
                if notebook_success:
                    successful_runs += 1
                    self.logger.info(f"Run {run_num} completed successfully")
                else:
                    failed_runs.append(run_num)
                    self.logger.error(f"Run {run_num} failed at notebook execution")
            else:
                failed_runs.append(run_num)
                self.logger.error(f"Run {run_num} failed at experiment execution")
        
        # Create overall summary
        overall_metadata.update({
            "end_time": datetime.now().isoformat(),
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate": successful_runs / self.num_runs
        })
        
        with open(self.base_output_dir / "overall_metadata.json", "w") as f:
            json.dump(overall_metadata, f, indent=2)
        
        # Create aggregated summary for LLM analysis
        self.create_aggregated_summary()
        
        self.logger.info(f"Completed all runs: {successful_runs}/{self.num_runs} successful")
        return successful_runs, failed_runs
    
    def create_aggregated_summary(self):
        """Create an aggregated summary of all runs for LLM analysis."""
        self.logger.info("Creating aggregated summary for LLM analysis")
        
        all_summaries = []
        
        # Collect all run summaries
        for run_num in range(1, self.num_runs + 1):
            summary_file = self.base_output_dir / f"run_{run_num}" / "summaries" / "run_summary.json"
            if summary_file.exists():
                try:
                    with open(summary_file) as f:
                        summary = json.load(f)
                        all_summaries.append(summary)
                except Exception as e:
                    self.logger.warning(f"Could not load summary for run {run_num}: {e}")
        
        # Create aggregated analysis
        aggregated = {
            "total_runs_analyzed": len(all_summaries),
            "overall_statistics": self.calculate_aggregate_stats(all_summaries),
            "individual_run_summaries": all_summaries,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Save aggregated data
        with open(self.base_output_dir / "aggregated_summary.json", "w") as f:
            json.dump(aggregated, f, indent=2)
        
        # Create text version for LLM analysis
        aggregated_text = self.format_aggregated_text(aggregated)
        with open(self.base_output_dir / "aggregated_summary.txt", "w") as f:
            f.write(aggregated_text)
        
        self.logger.info("Aggregated summary created")
    
    def calculate_aggregate_stats(self, summaries):
        """Calculate aggregate statistics across all runs."""
        if not summaries:
            return {
                "successful_runs": 0,
                "average_execution_time": 0,
                "std_execution_time": 0,
                "average_attack_success_rate": 0,
                "std_attack_success_rate": 0,
                "total_experiments_across_runs": 0,
                "average_experiments_per_run": 0
            }
        
        # Extract metrics from all runs
        execution_times = [s['run_metadata']['execution_time_seconds'] for s in summaries if s['run_metadata'].get('success', False)]
        attack_success_rates = [s['experiment_results'].get('attack_success_rate', 0) for s in summaries if s['experiment_results'].get('attack_success_rate') is not None]
        total_experiments = [s['experiment_results'].get('raw_experiments', 0) for s in summaries if s['experiment_results'].get('raw_experiments', 0) > 0]
        
        stats = {
            "successful_runs": sum(1 for s in summaries if s['run_metadata'].get('success', False)),
            "average_execution_time": np.mean(execution_times) if execution_times else 0,
            "std_execution_time": np.std(execution_times) if execution_times else 0,
            "average_attack_success_rate": np.mean(attack_success_rates) if attack_success_rates else 0,
            "std_attack_success_rate": np.std(attack_success_rates) if attack_success_rates else 0,
            "total_experiments_across_runs": sum(total_experiments),
            "average_experiments_per_run": np.mean(total_experiments) if total_experiments else 0
        }
        
        return stats
    
    def format_aggregated_text(self, aggregated):
        """Format aggregated summary for LLM analysis."""
        stats = aggregated['overall_statistics']
        
        text = f"""
MMCB Automated Experiment Analysis - Aggregated Summary
======================================================

Analysis Overview:
- Total Runs Completed: {aggregated['total_runs_analyzed']}
- Analysis Timestamp: {aggregated['analysis_timestamp']}

Overall Statistics:
- Successful Runs: {stats['successful_runs']}/{aggregated['total_runs_analyzed']}
- Average Execution Time: {stats['average_execution_time']:.1f} ± {stats['std_execution_time']:.1f} seconds
- Average Attack Success Rate: {stats['average_attack_success_rate']:.3f} ± {stats['std_attack_success_rate']:.3f}
- Total Experiments Across All Runs: {stats['total_experiments_across_runs']}
- Average Experiments Per Run: {stats['average_experiments_per_run']:.1f}

Individual Run Summary:
"""
        
        for i, summary in enumerate(aggregated['individual_run_summaries'], 1):
            run_meta = summary['run_metadata']
            exp_results = summary['experiment_results']
            
            text += f"""
Run {i}:
- Success: {run_meta['success']}
- Execution Time: {run_meta['execution_time_seconds']:.1f}s
- Attack Success Rate: {exp_results.get('attack_success_rate', 'N/A')}
- Total Experiments: {exp_results.get('raw_experiments', 'N/A')}
- Boundary Types: {', '.join(exp_results.get('boundary_types', []))}
- Attack Types: {', '.join(exp_results.get('attack_types', []))}
"""
        
        text += """

Instructions for LLM Analysis:
=============================

Please analyze the above data and provide:

1. Statistical Summary:
   - Overall trends across the 10 runs
   - Variance analysis and consistency metrics
   - Identification of outliers or anomalous runs

2. Performance Analysis:
   - Most effective boundary mechanisms (with confidence intervals)
   - Attack patterns and success rates
   - Cross-modal transfer effectiveness

3. Reliability Assessment:
   - Consistency of results across runs
   - Statistical significance of differences
   - Confidence in conclusions

4. Recommendations:
   - Based on aggregated data, which boundary mechanisms are most effective?
   - Are there implementation trade-offs that should be considered?
   - What are the key takeaways for defending against multi-modal attacks?

5. Future Work Suggestions:
   - Areas where additional experiments would be valuable
   - Potential improvements to the boundary mechanisms
   - Questions that remain unanswered

Please provide both a technical analysis and a summary suitable for a research paper.
"""
        
        return text

def main():
    """Main function to run the automated experiment suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="MMCB Automated Experiment Runner")
    parser.add_argument("--num_runs", type=int, default=25, help="Number of experimental runs")
    parser.add_argument("--output_dir", type=str, default="automated_runs", help="Base output directory")
    parser.add_argument("--skip_notebooks", action="store_true", help="Skip notebook execution (experiment data only)")
    
    args = parser.parse_args()
    
    # Create and run the automated suite
    runner = MMCBAutomatedRunner(
        base_output_dir=args.output_dir,
        num_runs=args.num_runs,
        skip_notebooks=args.skip_notebooks
    )
    
    try:
        successful_runs, failed_runs = runner.run_all_experiments()
        
        print(f"\n{'='*60}")
        print("AUTOMATED EXPERIMENT SUITE COMPLETED")
        print(f"{'='*60}")
        print(f"Successful runs: {successful_runs}/{args.num_runs}")
        print(f"Failed runs: {failed_runs}")
        print(f"Output directory: {args.output_dir}")
        print(f"Aggregated summary: {args.output_dir}/aggregated_summary.txt")
        print(f"{'='*60}")
        
        if successful_runs >= 5:  # At least half successful
            print("✅ Sufficient data collected for meaningful analysis")
            print(f"📄 Upload {args.output_dir}/aggregated_summary.txt to your LLM for analysis")
        else:
            print("⚠️  Limited data collected - consider investigating failures")
            
    except KeyboardInterrupt:
        print("\n🛑 Experiment suite interrupted by user")
    except Exception as e:
        print(f"❌ Experiment suite failed: {e}")
        raise

if __name__ == "__main__":
    main()