import os
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


class SLMService:
    def __init__(self):
        self.cache_dir = "./models/cache"
        self.base_model_dir = "./models/gemma-2b-base"
        self.finetuned_model_dir = "./models/gemma-2b-finetuned"
        self.hub_fallback = "google/gemma-2-2b-it"

        self.current_model_type = None
        self.model = None
        self.tokenizer = None

    def load_model(self, model_type: str = "finetuned"):
        """
        model_type: 'base' or 'finetuned'
        """
        if self.model is not None and self.current_model_type == model_type:
            return  # Already loaded in memory

        print(f"Loading local SLM model: '{model_type}'...")

        # Determine path to base model/weights
        base_path = self.base_model_dir if os.path.exists(self.base_model_dir) else self.hub_fallback

        # Load Tokenizer & Base Weights (Cached locally)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path,
            cache_dir=self.cache_dir,
            local_files_only=os.path.exists(self.base_model_dir)
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            cache_dir=self.cache_dir,
            local_files_only=os.path.exists(self.base_model_dir)
        )

        if model_type == "finetuned" and os.path.exists(self.finetuned_model_dir):
            print("Applying fine-tuned LoRA adapters...")
            self.model = PeftModel.from_pretrained(base_model, self.finetuned_model_dir)
        else:
            self.model = base_model

        self.current_model_type = model_type

    def generate_answer(self, query: str, model_type: str = "finetuned") -> str:
        self.load_model(model_type=model_type)

        # Prepend system instruction into the user prompt (Gemma does not support system role)
        system_instruction = "You are a helpful customer support assistant."
        combined_user_prompt = f"{system_instruction}\n\nCustomer Question: {query}"

        messages = [
            {"role": "user", "content": combined_user_prompt}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True
            )

        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return response_text.strip()

slm_service = SLMService()