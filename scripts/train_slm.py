import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer

# Directory configurations
CACHE_DIR = "./models/cache"
BASE_MODEL_SAVE_DIR = "./models/gemma-2b-base"
FINETUNED_MODEL_SAVE_DIR = "./models/gemma-2b-finetuned"
HUB_MODEL_ID = "google/gemma-2-2b-it"
DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"


def train_and_save():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(BASE_MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(FINETUNED_MODEL_SAVE_DIR, exist_ok=True)

    print("Step 1: Loading Tokenizer and Base Model with local caching...")
    tokenizer = AutoTokenizer.from_pretrained(
        HUB_MODEL_ID,
        cache_dir=CACHE_DIR,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Setup 4-bit Quantization Config (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        HUB_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )

    # Save Base Model & Tokenizer locally for inference without fine-tuning
    print(f"Step 2: Saving original base model to '{BASE_MODEL_SAVE_DIR}'...")
    model.save_pretrained(BASE_MODEL_SAVE_DIR)
    tokenizer.save_pretrained(BASE_MODEL_SAVE_DIR)

    # Prepare model for LoRA Training
    print("Step 3: Preparing model for LoRA fine-tuning...")
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)

    # Load dataset with cache enabled
    print(f"Step 4: Loading dataset '{DATASET_NAME}'...")
    dataset = load_dataset(DATASET_NAME, split="train[:1000]", cache_dir=CACHE_DIR)

    def format_prompts(examples):
        texts = []
        for instruction, response in zip(examples["instruction"], examples["response"]):
            messages = [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": response},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
        return {"text": texts}

    formatted_dataset = dataset.map(format_prompts, batched=True)

    # Define SFT Training Configuration
    training_args = SFTConfig(
        output_dir=FINETUNED_MODEL_SAVE_DIR,
        dataset_text_field="text",
        max_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
    )

    # 6. Execute Training
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("Step 5: Training Fine-Tuned Model...")
    trainer.train()

    print(f"Step 6: Saving Fine-Tuned model to '{FINETUNED_MODEL_SAVE_DIR}'...")
    trainer.model.save_pretrained(FINETUNED_MODEL_SAVE_DIR)
    tokenizer.save_pretrained(FINETUNED_MODEL_SAVE_DIR)
    print("Process complete! Base and Fine-Tuned models are saved and cached.")

if __name__ == "__main__":
    train_and_save()