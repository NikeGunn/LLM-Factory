from datasets import Dataset, load_dataset
import pandas as pd

class DatasetBuilder:
    def __init__(self, source_type="github"):
        self.source_type = source_type

    def load_from_github(self, repo_url=None):
        """Load sample code dataset"""
        return Dataset.from_pandas(pd.DataFrame({
            "text": [
                "def hello_world():\n    print('Hello World!')",
                "def add(a, b):\n    return a + b",
                "def factorial(n):\n    return 1 if n <= 1 else n * factorial(n-1)"
            ]
        }))

    def build(self, output_path="data/processed/dataset"):
        """Process and save dataset"""
        if self.source_type == "github":
            dataset = self.load_from_github()
        dataset.save_to_disk(output_path)
