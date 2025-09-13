"""
Configuration utilities for training.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
from dotenv import load_dotenv


def load_training_config() -> None:
    """
    Load environment variables from .env file.
    """
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        print(f"Warning: .env file not found at {env_path}")


def get_env_var(key: str, default: Optional[str] = None) -> str:
    """
    Get environment variable with optional default.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    value = os.getenv(key, default)
    if value is None:
        raise ValueError(f"Environment variable {key} is required but not set")
    return value


def get_env_bool(key: str) -> bool:
    """
    Get boolean environment variable.
    
    Args:
        key: Environment variable name
        
    Returns:
        Boolean value from environment
        
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Environment variable {key} is required but not set")
    return value.lower() in ("true", "1", "yes", "on")


def get_env_int(key: str) -> int:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable name
        
    Returns:
        Integer value from environment
        
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Environment variable {key} is required but not set")
    return int(value)


def get_env_float(key: str) -> float:
    """
    Get float environment variable.
    
    Args:
        key: Environment variable name
        
    Returns:
        Float value from environment
        
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if value is None:
        raise ValueError(f"Environment variable {key} is required but not set")
    return float(value)


def get_env_list(key: str) -> List[str]:
    """
    Get environment variable as comma-separated list.
    
    Args:
        key: Environment variable name
        
    Returns:
        List of strings from comma-separated environment variable
        
    Raises:
        ValueError: If environment variable is not set
    """
    value = os.getenv(key)
    if not value:
        raise ValueError(f"Environment variable {key} is required but not set")
    return [item.strip() for item in value.split(",") if item.strip()]


def get_env_dict(key: str, fallback_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Get environment variable as key:value dictionary.
    Supports quoted keys for names with special characters.
    
    Args:
        key: Environment variable name
        fallback_dict: Dictionary to use if environment variable is not set
        
    Returns:
        Dictionary parsed from key:value pairs in environment variable
        
    Raises:
        ValueError: If environment variable is not set and no fallback provided
    """
    value = os.getenv(key)
    if not value:
        if not fallback_dict:
            raise ValueError(f"Environment variable {key} is required but not set")
        return fallback_dict
    
    result = {}
    for item in value.split(","):
        if ":" in item:
            key_part, value_part = item.split(":", 1)
            
            # Remove quotes from key if present
            key_part = key_part.strip()
            if key_part.startswith('"') and key_part.endswith('"'):
                key_part = key_part[1:-1]
            elif key_part.startswith("'") and key_part.endswith("'"):
                key_part = key_part[1:-1]
            
            try:
                result[key_part] = float(value_part.strip())
            except ValueError:
                continue
    return result
