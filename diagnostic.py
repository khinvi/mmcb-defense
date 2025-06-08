#!/usr/bin/env python3
"""
MMCB Diagnostic Script - Check setup and troubleshoot issues
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description} NOT FOUND: {filepath}")
        return False

def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n🔍 Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print(f"✅ {description} - SUCCESS")
            if result.stdout.strip():
                print(f"Output: {result.stdout.strip()[:200]}")  # First 200 chars
        else:
            print(f"❌ {description} - FAILED (return code: {result.returncode})")
            print(f"Error: {result.stderr.strip()[:200]}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} - TIMEOUT")
        return False
    except FileNotFoundError:
        print(f"❌ {description} - COMMAND NOT FOUND")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False

def check_python_packages():
    """Check if required Python packages are installed."""
    packages = ['pandas', 'numpy', 'jupyter', 'nbconvert', 'matplotlib', 'seaborn']
    
    print("\n📦 Checking Python packages...")
    all_good = True
    
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
            all_good = False
    
    return all_good

def test_notebook_execution():
    """Test notebook execution with a simple example."""
    print("\n📓 Testing notebook execution...")
    
    # Create a simple test notebook
    test_notebook_content = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": ["# Test Notebook\n", "This is a simple test."]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": ["print('Hello from test notebook!')"]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {"name": "python", "version": "3.8.0"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    # Save test notebook
    test_nb_path = "test_notebook.ipynb"
    try:
        with open(test_nb_path, 'w') as f:
            json.dump(test_notebook_content, f, indent=2)
        
        print(f"Created test notebook: {test_nb_path}")
        
        # Try to execute it
        cmd = [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute", 
            "--ExecutePreprocessor.timeout=60",
            "--output", "test_notebook_executed.ipynb",
            test_nb_path
        ]
        
        success = run_command(cmd, "Test notebook execution")
        
        # Cleanup
        for cleanup_file in ["test_notebook.ipynb", "test_notebook_executed.ipynb"]:
            if Path(cleanup_file).exists():
                os.remove(cleanup_file)
                
        return success
        
    except Exception as e:
        print(f"❌ Test notebook creation failed: {e}")
        return False

def check_mmcb_setup():
    """Check MMCB project setup."""
    print("\n🔬 Checking MMCB project setup...")
    
    files_to_check = [
        ("src/main.py", "Main experiment script"),
        ("notebooks/analysis.ipynb", "Analysis notebook"), 
        ("config/experiment.yaml", "Experiment configuration"),
        ("requirements.txt", "Requirements file")
    ]
    
    all_files_exist = True
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_files_exist = False
    
    return all_files_exist

def diagnose_notebook_failure():
    """Provide specific guidance for notebook execution failures."""
    print("\n🩺 Notebook Execution Failure Diagnosis:")
    print("=" * 50)
    
    # Check if analysis.ipynb actually exists and has content
    notebook_path = "notebooks/analysis.ipynb"
    if Path(notebook_path).exists():
        try:
            with open(notebook_path, 'r') as f:
                nb_content = json.load(f)
            
            num_cells = len(nb_content.get('cells', []))
            print(f"📊 Notebook has {num_cells} cells")
            
            # Check for common problematic patterns
            code_cells = [cell for cell in nb_content.get('cells', []) if cell.get('cell_type') == 'code']
            print(f"📊 Found {len(code_cells)} code cells")
            
            # Look for cells with results_df
            data_loading_cells = []
            for i, cell in enumerate(code_cells):
                source = ''.join(cell.get('source', []))
                if 'results_df' in source or 'checkpoint' in source.lower() or 'find_latest' in source:
                    data_loading_cells.append(i + 1)
            
            if data_loading_cells:
                print(f"📊 Found data loading cells at positions: {data_loading_cells}")
                print("💡 LIKELY ISSUE: Notebook expects existing experiment data")
                print("💡 SOLUTION: Run experiments first, then notebooks will work")
            else:
                print("🤔 No obvious data dependencies found")
                
        except json.JSONDecodeError:
            print("❌ Notebook file is corrupted (invalid JSON)")
        except Exception as e:
            print(f"❌ Error reading notebook: {e}")
    
    print("\n💡 Common Solutions:")
    print("1. Skip notebook execution for now:")
    print("   python automated_runner.py --skip_notebooks")
    print("2. Fix notebook data dependencies")
    print("3. Use simpler notebook or create test version")

def main():
    """Run all diagnostic checks."""
    print("🔧 MMCB Automated Runner Diagnostic Tool")
    print("=" * 50)
    
    # Check basic setup
    python_ok = check_python_packages()
    files_ok = check_mmcb_setup()
    
    # Check external commands
    jupyter_ok = run_command(["jupyter", "--version"], "Jupyter installation")
    nbconvert_ok = run_command(["jupyter", "nbconvert", "--help"], "nbconvert functionality")
    
    # Test notebook execution
    notebook_test_ok = test_notebook_execution()
    
    # Summary
    print("\n📋 DIAGNOSTIC SUMMARY")
    print("=" * 30)
    
    checks = [
        ("Python packages", python_ok),
        ("MMCB project files", files_ok), 
        ("Jupyter command", jupyter_ok),
        ("nbconvert command", nbconvert_ok),
        ("Notebook execution test", notebook_test_ok)
    ]
    
    all_passed = True
    for check_name, status in checks:
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {check_name}")
        if not status:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All checks passed! You should be able to run the automated suite.")
        print("Run: python automated_runner.py --num_runs 5")
    else:
        print("\n⚠️  Some issues detected. See recommendations below:")
        
        if not python_ok:
            print("📦 Install missing packages: pip install pandas numpy jupyter nbconvert matplotlib seaborn")
        
        if not jupyter_ok:
            print("🔧 Install Jupyter: pip install jupyter")
            
        if not nbconvert_ok:
            print("🔧 Install nbconvert: pip install nbconvert")
            
        if not notebook_test_ok:
            diagnose_notebook_failure()
            print("\n🚀 QUICK FIX: Run with notebook execution disabled:")
            print("python automated_runner.py --num_runs 5 --skip_notebooks")
            
        if not files_ok:
            print("📁 Ensure you're running from the MMCB project root directory")

if __name__ == "__main__":
    main()