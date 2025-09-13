"""
Configuration utilities for training.
"""

import os
from pathlib import Path
from typing import Optional
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


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable.
    
    Args:
        key: Environment variable name
        default: Default boolean value
        
    Returns:
        Boolean value from environment
    """
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


def get_env_int(key: str, default: Optional[int] = None) -> int:
    """
    Get integer environment variable.
    
    Args:
        key: Environment variable name
        default: Default integer value
        
    Returns:
        Integer value from environment
    """
    value = os.getenv(key)
    if value is None:
        if default is None:
            raise ValueError(f"Environment variable {key} is required but not set")
        return default
    return int(value)


def get_env_float(key: str, default: Optional[float] = None) -> float:
    """
    Get float environment variable.
    
    Args:
        key: Environment variable name
        default: Default float value
        
    Returns:
        Float value from environment
    """
    value = os.getenv(key)
    if value is None:
        if default is None:
            raise ValueError(f"Environment variable {key} is required but not set")
        return default
    return float(value)
