#!/bin/bash

# Initialize repository
mkdir -p LLM-Factory/{.github/workflows,configs,data/{raw,processed},models/{base_models,trained_models},notebooks,scripts,src/{data_loader,trainer,inference,utils},tests,docs}

# Create core files
touch LLM-Factory/{.gitignore,Dockerfile,requirements.txt}

# .github files
cat > LLM-Factory/.github/workflows/deploy.yml << 'EOL'
name: Deploy API
on: push
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: bash scripts/setup_env.sh
      - run: bash scripts/deploy_api.sh
EOL

cat > LLM-Factory/.github/workflows/tests.yml << 'EOL'
name: Run Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - run: pip install -r requirements.txt
      - run: pytest tests/
EOL

# Config files
cat > LLM-Factory/configs/train_config.yaml << 'EOL'
model_name: "mistralai/Mistral-7B-v0.1"
dataset_path: "data/processed/code_dataset"
lora_rank: 8
lora_alpha: 32
target_modules: ["q_proj", "v_proj"]
batch_size: 4
epochs: 3
learning_rate: 2e-5
EOL

cat > LLM-Factory/configs/model_config.yaml << 'EOL'
quantization: "4bit"
flash_attention: true
device_map: "auto"
EOL

# Data processing
cat > LLM-Factory/data/custom_dataset.py << 'EOL'
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
EOL

# Training code
mkdir -p LLM-Factory/src/trainer
cat > LLM-Factory/src/trainer/lora_trainer.py << 'EOL'
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml
from datasets import load_from_disk
import torch

def train():
    # Load config
    with open("../../configs/train_config.yaml") as f:
        config = yaml.safe_load(f)

    # Load model with 4-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # Prepare for LoRA training
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"]
    )
    model = get_peft_model(model, peft_config)

    # Load dataset
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    dataset = load_from_disk(config["dataset_path"])

    # Training setup
    trainer = Trainer(
        model=model,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir="../../models/trained_models",
            per_device_train_batch_size=config["batch_size"],
            num_train_epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            fp16=True,
            logging_steps=10,
            save_strategy="steps"
        ),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )

    # Start training
    trainer.train()
    model.save_pretrained("../../models/trained_models")
EOL

# Inference code
cat > LLM-Factory/src/inference/server.py << 'EOL'
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware
import torch

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        "../../models/trained_models",
        device_map="auto",
        torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained("../../models/trained_models")

@app.post("/generate")
async def generate(prompt: str, max_length: int = 200):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9
    )
    return {"response": tokenizer.decode(outputs[0], skip_special_tokens=True)}
EOL

# Utils
cat > LLM-Factory/src/utils/logger.py << 'EOL'
import logging

def setup_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    return logger
EOL

# Notebooks
cat > LLM-Factory/notebooks/1-Finetuning-Guide.ipynb << 'EOL'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# LLM Fine-Tuning Guide"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!pip install -q transformers peft datasets"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from src.trainer.lora_trainer import train\n",
    "train()"
   ]
  }
 ]
}
EOL

# Scripts
cat > LLM-Factory/scripts/setup_env.sh << 'EOL'
#!/bin/bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
EOL

cat > LLM-Factory/scripts/train_model.sh << 'EOL'
#!/bin/bash
python -m src.trainer.lora_trainer
EOL

cat > LLM-Factory/scripts/deploy_api.sh << 'EOL'
#!/bin/bash
uvicorn src.inference.server:app --host 0.0.0.0 --port 8000 --reload
EOL

# Make scripts executable
chmod +x LLM-Factory/scripts/*.sh

# Requirements
cat > LLM-Factory/requirements.txt << 'EOL'
torch>=2.0.1
transformers>=4.31.0
peft>=0.4.0
bitsandbytes>=0.40.0
accelerate>=0.21.0
datasets>=2.14.0
fastapi>=0.95.0
uvicorn>=0.22.0
pytest>=7.4.0
python-dotenv>=1.0.0
EOL

# Dockerfile
cat > LLM-Factory/Dockerfile << 'EOL'
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
CMD ["bash", "scripts/deploy_api.sh"]
EOL

# README.md
cat > LLM-Factory/README.md << 'EOL'
# 🏭 LLM Factory

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

The complete solution for building and deploying custom LLMs.

## 🚀 Features

- One-click fine-tuning with LoRA
- Pre-configured for Mistral/Llama2
- FastAPI + vLLM deployment
- CI/CD ready

## 📂 Folder Structure
LLM-Factory/
├── configs/ # Training configurations
├── data/ # Dataset processing
├── models/ # Model storage
├── notebooks/ # Tutorial notebooks
├── scripts/ # One-click scripts
└── src/ # Core Python modules
