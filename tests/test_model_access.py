# Save as test_model_access.py
from huggingface_hub import HfApi
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_model_access():
    api = HfApi()
    
    # Test authentication
    try:
        user_info = api.whoami()
        print(f"✅ Authenticated as: {user_info['name']}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return False
    
    # Test model access
    models_to_test = [
        "meta-llama/Llama-3-8b-hf",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    
    for model_name in models_to_test:
        try:
            print(f"Testing access to {model_name}...")
            
            # Try to load tokenizer (lightweight test)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            print(f"✅ {model_name}: Access granted")
            
        except Exception as e:
            if "gated" in str(e) or "401" in str(e):
                print(f"❌ {model_name}: Access not granted yet - request access on HuggingFace")
            else:
                print(f"❌ {model_name}: Other error - {e}")
    
    return True

if __name__ == "__main__":
    test_model_access()