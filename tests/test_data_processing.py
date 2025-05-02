from llm_factory.data_loader.dataset_loader import load_and_prepare_dataset
from transformers import AutoTokenizer

def test_data_loading():
    """Test that dataset loads without errors"""
    # Use a small test model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    try:
        dataset = load_and_prepare_dataset(tokenizer, max_length=128)
        assert dataset is not None
    except Exception as e:
        pytest.fail(f"Data loading failed with {str(e)}")