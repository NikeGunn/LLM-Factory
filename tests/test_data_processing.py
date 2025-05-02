from src.data_loader.dataset_loader import load_and_prepare_dataset
from transformers import AutoTokenizer

def test_data_loading():
    """Test that dataset loads without errors"""
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    dataset = load_and_prepare_dataset(tokenizer, max_length=128)
    assert len(dataset) > 0, "Dataset should not be empty"