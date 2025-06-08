from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, validator, root_validator, Field, field_validator

SUPPORTED_MODELS = [
    'llama3-8b', 'mistral-7b', 'Llama-3-8B', 'Mistral-7B',
    'meta-llama/Meta-Llama-3-8B-Instruct', 'mistralai/Mistral-7B-Instruct-v0.2'
]
SUPPORTED_BOUNDARIES = ['token', 'semantic', 'hybrid']
SUPPORTED_ATTACK_TYPES = ['json', 'csv', 'yaml', 'xml', 'python', 'javascript']

class ModelConfig(BaseModel):
    name: str
    device: Optional[str] = Field(None, description="Device to use: cpu, cuda, or mps.")
    path: Optional[str] = None
    # Add more model-specific fields as needed

    @validator('name')
    def name_supported(cls, v):
        if v not in SUPPORTED_MODELS:
            raise ValueError(f"Model name '{v}' not supported. Supported: {SUPPORTED_MODELS}")
        return v
    @validator('device')
    def device_valid(cls, v):
        if v is not None and v not in ['cpu', 'cuda', 'mps']:
            raise ValueError(f"Device '{v}' not valid. Must be one of: cpu, cuda, mps.")
        return v

class ExperimentConfig(BaseModel):
    models: List[Union[str, ModelConfig]]
    boundaries: List[str]
    attack_types: List[str]
    num_attacks: int
    batch_size: int
    checkpoint_interval: int
    experiment: Optional[Dict[str, Any]] = None
    extends: Optional[str] = None

    @validator('models', each_item=True)
    def model_entry_valid(cls, v):
        if isinstance(v, str):
            if v not in SUPPORTED_MODELS:
                raise ValueError(f"Model name '{v}' not supported. Supported: {SUPPORTED_MODELS}")
        elif isinstance(v, dict):
            ModelConfig(**v)  # Will raise if invalid
        elif isinstance(v, ModelConfig):
            pass
        else:
            raise ValueError("Each model entry must be a string or ModelConfig.")
        return v

    @validator('boundaries', each_item=True)
    def boundary_supported(cls, v):
        if v not in SUPPORTED_BOUNDARIES:
            raise ValueError(f"Boundary type '{v}' not supported. Supported: {SUPPORTED_BOUNDARIES}")
        return v

    @validator('attack_types', each_item=True)
    def attack_type_supported(cls, v):
        if v not in SUPPORTED_ATTACK_TYPES:
            raise ValueError(f"Attack type '{v}' not supported. Supported: {SUPPORTED_ATTACK_TYPES}")
        return v

    @field_validator('num_attacks', 'batch_size', 'checkpoint_interval')
    def positive_int(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("Value must be a positive integer.")
        return v

    class Config:
        extra = 'allow'  # Allow extra fields for forward compatibility
        json_schema_extra = {
            "example": {
                "models": ["llama3-8b", {"name": "mistral-7b", "device": "cuda"}],
                "boundaries": ["token", "semantic"],
                "attack_types": ["json", "csv"],
                "num_attacks": 10,
                "batch_size": 2,
                "checkpoint_interval": 5,
                "experiment": {"description": "Test run"}
            }
        } 