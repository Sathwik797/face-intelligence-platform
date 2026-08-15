import os
import yaml
from typing import Any, Dict

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """Loads YAML configuration file."""
    if not os.path.exists(config_path):
        # Fallback to relative path if called from different working directory
        if os.path.exists("config/config.yaml"):
            config_path = "config/config.yaml"
        else:
            raise FileNotFoundError(f"Configuration file not found at {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
