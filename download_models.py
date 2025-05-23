#!/usr/bin/env python3
"""
Download script for MMCB models on M3 Pro
Run this to pre-download models before running experiments
Includes authentication handling for Llama models
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from huggingface_hub import login, HfApi
import subprocess
import sys

def setup_cache_dir():
    """Set up cache directory for models."""
    cache_dir = os.path.expanduser("~/models/huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir

def check_huggingface_auth():
    """Check if user is authenticated with Hugging Face."""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
        return True
    except Exception:
        print("❌ Not authenticated with Hugging Face")
        return False

def setup_huggingface_auth():
    """Set up Hugging Face authentication."""
    print("\n🔐 Hugging Face Authentication Required")
    print("=" * 50)
    print("Some models (like Llama) require authentication.")
    print("You need to:")
    print("1. Create account at https://huggingface.co")
    print("2. Accept model licenses (especially for Llama models)")
    print("3. Create access token at https://huggingface.co/settings/tokens")
    print()
    
    choice = input("Do you want to login now? (y/n): ").lower()
    if choice != 'y':
        print("Skipping authentication. Some models may fail to download.")
        return False
    
    try:
        # Try to install huggingface_hub if not available
        try:
            from huggingface_hub import login
        except ImportError:
            print("Installing huggingface_hub...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
            from huggingface_hub import login
        
        # Prompt for login
        print("\nPlease enter your Hugging Face token:")
        print("(Get it from: https://huggingface.co/settings/tokens)")
        token = input("Token: ").strip()
        
        if token:
            login(token=token)
            print("✅ Successfully authenticated!")
            return True
        else:
            print("❌ No token provided")
            return False
            
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False

def download_model(model_name, cache_dir, use_auth=False):
    """Download a single model with optional authentication."""
    print(f"Downloading {model_name}...")
    print(f"This may take 10-30 minutes depending on your internet speed.")
    
    try:
        # Prepare auth token parameter
        auth_kwargs = {}
        if use_auth:
            auth_kwargs['use_auth_token'] = True
        
        # Download tokenizer first (small)
        print(f"  Downloading tokenizer for {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            cache_dir=cache_dir,
            **auth_kwargs
        )
        
        # Download model (large)
        print(f"  Downloading model weights for {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Optimized for M3 Pro
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            **auth_kwargs
        )
        
        print(f"✅ Successfully downloaded {model_name}")
        
        # Clean up memory
        del model
        del tokenizer
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        
        # Provide helpful error messages
        if "401" in str(e) or "authentication" in str(e).lower():
            print("   This appears to be an authentication error.")
            print("   Make sure you've accepted the model license and have a valid token.")
        elif "not found" in str(e).lower():
            print("   This model might not exist or the name has changed.")
            print("   Please check the model name on https://huggingface.co")
        
        return False
    
    return True

def get_model_list():
    """Get list of models with their authentication requirements."""
    return [
        {
            "name": "mistralai/Mistral-7B-Instruct-v0.2",
            "size": "~14GB",
            "auth_required": False,
            "description": "Mistral 7B - No authentication required"
        },
        {
            "name": "meta-llama/Meta-Llama-3-8B-Instruct", 
            "size": "~16GB",
            "auth_required": True,
            "description": "Llama 3 8B - Requires authentication",
            "alternatives": [
                "meta-llama/Llama-2-7b-chat-hf",
                "meta-llama/Meta-Llama-3-8B"
            ]
        }
    ]

def main():
    """Download all models for the MMCB project."""
    print("MMCB Model Download Script")
    print("=" * 40)
    
    # Check system
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")
    print()
    
    # Set up cache directory
    cache_dir = setup_cache_dir()
    print(f"Cache directory: {cache_dir}")
    print()
    
    # Get model list
    models = get_model_list()
    
    # Display model information
    print("Models to download:")
    total_size = 0
    for i, model in enumerate(models, 1):
        auth_status = "🔐 Auth required" if model["auth_required"] else "🔓 Public"
        print(f"{i}. {model['name']} ({model['size']}) - {auth_status}")
        if 'alternatives' in model:
            print(f"   Alternatives: {', '.join(model['alternatives'])}")
        # Rough size calculation
        size_gb = int(model['size'].replace('~', '').replace('GB', ''))
        total_size += size_gb
    
    print(f"\nTotal download size: ~{total_size}GB")
    print()
    
    # Check authentication status
    auth_available = check_huggingface_auth()
    
    # If authentication is needed but not available, offer to set it up
    auth_required_models = [m for m in models if m["auth_required"]]
    if auth_required_models and not auth_available:
        auth_available = setup_huggingface_auth()
    
    # Confirm download
    response = input(f"Continue with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled.")
        return
    
    # Download each model
    success_count = 0
    failed_models = []
    
    for model_info in models:
        model_name = model_info["name"]
        use_auth = model_info["auth_required"]
        
        # Skip auth-required models if no auth available
        if use_auth and not auth_available:
            print(f"⏭️  Skipping {model_name} (authentication required)")
            failed_models.append(model_name)
            continue
        
        # Try to download the model
        if download_model(model_name, cache_dir, use_auth=use_auth):
            success_count += 1
        else:
            failed_models.append(model_name)
            
            # Try alternatives for Llama if available
            if 'alternatives' in model_info:
                print(f"   Trying alternatives for {model_name}...")
                for alt_model in model_info['alternatives']:
                    print(f"   Attempting: {alt_model}")
                    if download_model(alt_model, cache_dir, use_auth=use_auth):
                        success_count += 1
                        failed_models.remove(model_name)  # Remove from failed list
                        print(f"   ✅ Successfully downloaded alternative: {alt_model}")
                        
                        # Update your config to use this alternative
                        print(f"   💡 Update your config to use: {alt_model}")
                        break
        print()
    
    # Final summary
    print("=" * 40)
    print(f"Download complete: {success_count}/{len(models)} models downloaded")
    
    if success_count == len(models):
        print("✅ All models ready for MMCB experiments!")
    elif success_count > 0:
        print(f"✅ {success_count} model(s) ready for experiments!")
        if failed_models:
            print(f"⚠️  Failed models: {', '.join(failed_models)}")
            print("   You can run experiments with the downloaded models.")
    else:
        print("❌ No models downloaded successfully.")
        print("   Check your internet connection and authentication.")
    
    # Provide next steps
    if success_count > 0:
        print("\n🚀 Next steps:")
        print("1. Update your config/models.yaml with the downloaded models")
        print("2. Test with: python src/main.py --quick")
        print("3. Run full experiments when ready!")

def test_downloaded_models():
    """Test function to verify downloaded models work."""
    print("\n🧪 Testing downloaded models...")
    
    models_to_test = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "meta-llama/Llama-2-7b-chat-hf"
    ]
    
    for model_name in models_to_test:
        try:
            print(f"Testing {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Just load tokenizer to verify model exists
            print(f"✅ {model_name} - Ready!")
            
        except Exception as e:
            print(f"❌ {model_name} - Not available: {e}")

if __name__ == "__main__":
    main()
    
    # Optionally test the models
    test_choice = input("\nTest downloaded models? (y/n): ")
    if test_choice.lower() == 'y':
        test_downloaded_models()