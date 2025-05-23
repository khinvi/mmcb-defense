#!/usr/bin/env python3
"""
Quick Model Test - Fast verification that models work
Run this first for a quick check
"""

import sys
import time
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

def quick_test():
    """Quick test of both models."""
    print("🚀 Quick Model Test")
    print("=" * 30)
    
    # Check PyTorch and MPS
    print(f"PyTorch: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    try:
        from models.mistral import MistralModel
        from models.llama import LlamaModel
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    
    # Test Mistral (smaller, loads faster)
    print("\n🤖 Testing Mistral...")
    try:
        start = time.time()
        mistral = MistralModel("mistralai/Mistral-7B-Instruct-v0.2")
        load_time = time.time() - start
        
        response = mistral.generate_response("Hello!", max_tokens=20)
        print(f"✅ Mistral working! ({load_time:.1f}s load)")
        print(f"   Response: {response[:50]}...")
        
    except Exception as e:
        print(f"❌ Mistral failed: {e}")
        return False
    
    # Test Llama
    print("\n🦙 Testing Llama...")
    try:
        start = time.time()
        llama = LlamaModel("meta-llama/Meta-Llama-3-8B-Instruct")
        load_time = time.time() - start
        
        response = llama.generate_response("Hello!", max_tokens=20)
        print(f"✅ Llama working! ({load_time:.1f}s load)")
        print(f"   Response: {response[:50]}...")
        
    except Exception as e:
        print(f"❌ Llama failed: {e}")
        return False
    
    print("\n🎉 Both models working perfectly!")
    return True

if __name__ == "__main__":
    if quick_test():
        print("\n✅ Ready to run: python src/main.py --quick")
    else:
        print("\n❌ Fix issues before running experiments")