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
