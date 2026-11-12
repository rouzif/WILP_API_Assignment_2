import os
import torch
import json
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.services.rag import rag_service


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
        if self.model is not None and self.current_model_type == model_type:
            return

        base_path = self.base_model_dir if os.path.exists(self.base_model_dir) else self.hub_fallback

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
            self.model = PeftModel.from_pretrained(base_model, self.finetuned_model_dir)
        else:
            self.model = base_model

        self.current_model_type = model_type

    def generate_text(self, prompt_content: str) -> str:
        messages = [{"role": "user", "content": prompt_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.2,
                do_sample=True
            )

        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()

    def process_query_workflow(
            self,
            query: str,
            model_type: str = "finetuned",
            use_rag: bool = False,
            domain: str = "vanilla",
            chat_history: list = None
    ):
        self.load_model(model_type=model_type)

        context = ""
        if use_rag:
            context = rag_service.retrieve_context(query, domain=domain)

        # Sentiment & Escalation Analysis
        eval_prompt = (
            f"Analyze this query: \"{query}\"\n"
            "Return valid JSON only with keys:\n"
            "\"sentiment\": (\"Positive\", \"Neutral\", or \"Negative\")\n"
            "\"human_intervention_required\": (true or false)"
        )

        raw_eval = self.generate_text(eval_prompt)

        sentiment = "Neutral"
        needs_human = False
        try:
            cleaned_json = raw_eval[raw_eval.find("{"):raw_eval.rfind("}") + 1]
            parsed = json.loads(cleaned_json)
            sentiment = parsed.get("sentiment", "Neutral")
            needs_human = bool(parsed.get("human_intervention_required", False))
        except Exception:
            if "negative" in raw_eval.lower():
                sentiment = "Negative"
            elif "positive" in raw_eval.lower():
                sentiment = "Positive"
            needs_human = "true" in raw_eval.lower()

        # Multi-turn prompt helper
        def build_response_prompt(instruction: str) -> str:
            history_str = ""
            if chat_history:
                history_str = "Prior Conversation History:\n" + "\n".join(
                    [f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in chat_history[-6:]]
                ) + "\n\n"

            if context:
                return f"{instruction}\n\nOfficial Policy Context:\n{context}\n\n{history_str}Customer Query: {query}"
            return f"{instruction}\n\n{history_str}Customer Query: {query}"

        if sentiment in ["Positive", "Neutral"] and needs_human:
            answer = "Thank you for reaching out!. We are routing your query to a human chat agent right away."
        elif sentiment in ["Positive", "Neutral"] and not needs_human:
            gen_prompt = build_response_prompt(
                "Thank the customer and answer their query accurately using the provided policy context.")
            answer = self.generate_text(gen_prompt)
        elif sentiment == "Negative" and needs_human:
            answer = "We sincerely apologize for the inconvenience caused. We are immediately routing your query to an account manager."
        elif sentiment == "Negative" and not needs_human:
            gen_prompt = build_response_prompt(
                "Apologize to the customer for their issue and provide a solution using the provided policy context.")
            answer = self.generate_text(gen_prompt)
        else:
            gen_prompt = build_response_prompt(
                "Answer the customer query clearly and helpfully using the provided policy context.")
            answer = self.generate_text(gen_prompt)

        return {
            "sentiment": sentiment,
            "human_intervention_required": needs_human,
            "answer": answer
        }

slm_service = SLMService()