import yaml
from pathlib import Path

def load_config(config_path: str = "configs/config.yaml") -> dict:
    config_path = Path(__file__).resolve().parent.parent / config_path
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config