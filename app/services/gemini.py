import json
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, before_sleep_log
from app.config import settings
from app.schemas import QueryAnalysis, RouterDecision
from app.services.rag import rag_service

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self):
        self.api_key = settings.GEMINI_API_KEY
        self.base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=6),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _post_gemini_rest(self, model: str, prompt: str, json_schema: dict = None, temperature: float = 0.0) -> str:
        url = f"{self.base_url}/{model}:generateContent?key={self.api_key}"

        generation_config = {"temperature": temperature}
        if json_schema:
            generation_config["responseMimeType"] = "application/json"
            generation_config["responseSchema"] = json_schema

        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": generation_config
        }

        response = requests.post(
            url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=(15, 45)
        )
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def classify_complexity(self, query: str, model: str = settings.DEFAULT_MODEL) -> str:
        prompt = (
            f"Analyze this customer support query:\n\"{query}\"\n\n"
            "Classify it as either 'Simple' or 'Complex'.\n"
            "- 'Simple': Basic standard queries (e.g., store hours, standard return timeframe, basic shipping fee, simple greetings, straightforward how-to questions).\n"
            "- 'Complex': Multi-part questions, edge cases, complaints requiring nuanced judgment, or policy exceptions."
        )
        schema = {
            "type": "OBJECT",
            "properties": {
                "complexity": {"type": "STRING", "enum": ["Simple", "Complex"]}
            },
            "required": ["complexity"]
        }

        try:
            raw_text = self._post_gemini_rest(model=model, prompt=prompt, json_schema=schema, temperature=0.0)
            decision = RouterDecision.model_validate_json(raw_text)
            return decision.complexity
        except Exception as e:
            logger.error(f"[Gemini Classify Complexity Failed]: {str(e)}")
            return "Complex"

    def analyze_query(self, query: str, model: str = settings.DEFAULT_MODEL) -> dict:
        """Always executes Subtask 1 (Sentiment) & Subtask 2 (Escalation) via Gemini."""
        analysis_prompt = (
            f"Analyze this customer support query: \"{query}\"\n"
            "Rules:\n"
            "- sentiment: Classify as 'Positive', 'Neutral', or 'Negative'.\n"
            "- human_intervention_required: Set to true ONLY if the customer is angry, demands a human/manager, "
            "or reports an urgent account failure/billing dispute. Set to false for routine 'how-to' or policy questions."
        )
        schema = {
            "type": "OBJECT",
            "properties": {
                "sentiment": {"type": "STRING", "enum": ["Positive", "Neutral", "Negative"]},
                "human_intervention_required": {"type": "BOOLEAN"}
            },
            "required": ["sentiment", "human_intervention_required"]
        }

        raw_text = self._post_gemini_rest(model=model, prompt=analysis_prompt, json_schema=schema, temperature=0.0)
        analysis = QueryAnalysis.model_validate_json(raw_text)

        return {
            "sentiment": analysis.sentiment,
            "human_intervention_required": analysis.human_intervention_required
        }

    def process_query_workflow(
            self,
            query: str,
            model: str = settings.DEFAULT_MODEL,
            use_rag: bool = False,
            domain: str = "vanilla",
            chat_history: list = None,
            sentiment: str = None,
            needs_human: bool = None
    ):
        # Fallback if called directly without pre-computed Subtask 1 & 2 results
        if sentiment is None or needs_human is None:
            analysis = self.analyze_query(query=query, model=model)
            sentiment = analysis["sentiment"]
            needs_human = analysis["human_intervention_required"]

        context = ""
        if use_rag:
            context = rag_service.retrieve_context(query, domain=domain)

        # Subtask 3 Response Generation
        if sentiment == "Positive" and needs_human:
            answer = "Thank you! We are routing your query to a human chat agent."
        elif sentiment == "Negative" and needs_human:
            answer = "We sincerely apologize for your experience. We are routing your issue to an account manager."
        elif sentiment == "Neutral" and needs_human:
            answer = "I am routing your request to a human support specialist who can assist you further."
        else:
            history_str = ""
            if chat_history:
                recent_history = chat_history[-4:]
                history_str = "Prior Conversation History:\n" + "\n".join(
                    [f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in recent_history]
                ) + "\n\n"

            if use_rag and context:
                prompt = (
                    "You are a helpful customer support agent maintaining a continuous conversation.\n"
                    "STRICT INSTRUCTIONS: Answer the customer's question strictly using ONLY the provided policy context below. "
                    "Do NOT invent facts or placeholders. If the answer is not contained in the context, say 'I do not have information on that policy.'\n\n"
                    f"Official Policy Context:\n{context}\n\n"
                    f"{history_str}"
                    f"Customer Question: {query}"
                )
            else:
                prompt = f"{history_str}Customer Query: {query}\nProvide a helpful customer support response."

            answer = self._post_gemini_rest(model=model, prompt=prompt, temperature=0.0).strip()

        return {
            "sentiment": sentiment,
            "human_intervention_required": needs_human,
            "answer": answer
        }


gemini_service = GeminiService()