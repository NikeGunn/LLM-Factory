import yaml
from pathlib import Path

def test_config_files_exist():
    """Verify all config files exist"""
    assert Path("configs/train_config.yaml").exists()
    assert Path("configs/model_config.yaml").exists()

def test_config_contents():
    """Test config files are valid YAML"""
    with open("configs/train_config.yaml") as f:
        train_config = yaml.safe_load(f)
    assert "model_name" in train_config