import os
import torch
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
        device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            base_path,
            cache_dir=self.cache_dir,
            local_files_only=os.path.exists(self.base_model_dir)
        )

        base_model = AutoModelForCausalLM.from_pretrained(
            base_path,
            torch_dtype=dtype,
            cache_dir=self.cache_dir,
            local_files_only=os.path.exists(self.base_model_dir)
        ).to(device)

        if model_type == "finetuned" and os.path.exists(self.finetuned_model_dir):
            self.model = PeftModel.from_pretrained(base_model, self.finetuned_model_dir).to(device)
        else:
            self.model = base_model

        self.model.eval()
        self.current_model_type = model_type

    def generate_text(self, prompt_content: str) -> str:
        messages = [{"role": "user", "content": prompt_content}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
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
        chat_history: list = None,
        sentiment: str = "Neutral",
        needs_human: bool = False
    ):
        # Immediate escalation routing without loading local SLM weights
        if sentiment == "Positive" and needs_human:
            return {
                "sentiment": sentiment,
                "human_intervention_required": needs_human,
                "answer": "Thank you for reaching out! We appreciate your positive feedback. We are routing your query to a human chat agent right away."
            }
        elif sentiment == "Negative" and needs_human:
            return {
                "sentiment": sentiment,
                "human_intervention_required": needs_human,
                "answer": "We sincerely apologize for the inconvenience caused. We are immediately routing your query to an account manager."
            }
        elif sentiment == "Neutral" and needs_human:
            return {
                "sentiment": sentiment,
                "human_intervention_required": needs_human,
                "answer": "I am routing your request to a human support specialist who can assist you further."
            }

        # Load SLM only when generating a Subtask 3 response
        self.load_model(model_type=model_type)

        context = ""
        if use_rag:
            context = rag_service.retrieve_context(query, domain=domain)

        def build_response_prompt(instruction: str) -> str:
            strict_rule = (
                "STRICT RULE: Use ONLY the facts in the policy context below. "
                "Do NOT invent contact details, phone numbers, URLs, or bracketed placeholders."
            )
            history_str = ""
            if chat_history:
                recent_history = chat_history[-4:]
                history_str = "Prior Conversation History:\n" + "\n".join(
                    [f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in recent_history]
                ) + "\n\n"

            if context:
                return f"{instruction}\n{strict_rule}\n\nOfficial Policy Context:\n{context}\n\n{history_str}Customer Query: {query}"
            return f"{instruction}\n{strict_rule}\n\n{history_str}Customer Query: {query}"

        if sentiment == "Positive":
            gen_prompt = build_response_prompt("Thank the customer and answer their query accurately using the provided policy context.")
        elif sentiment == "Negative":
            gen_prompt = build_response_prompt("Apologize to the customer for their issue and provide a solution using the provided policy context.")
        else:
            gen_prompt = build_response_prompt("Answer the customer query clearly and helpfully using the provided policy context.")

        answer = self.generate_text(gen_prompt)

        return {
            "sentiment": sentiment,
            "human_intervention_required": needs_human,
            "answer": answer
        }


slm_service = SLMService()