import json
import time
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.config import settings
from app.schemas import QueryAnalysis, RouterDecision
from app.services.rag import rag_service


class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    # Retry helper for Gemini API calls: retries twice (3 attempts max) with exponential backoff
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=6),
        reraise=True
    )
    def _call_gemini_with_retry(self, model: str, contents: str, config: types.GenerateContentConfig):
        return self.client.models.generate_content(
            model=model,
            contents=contents,
            config=config
        )

    def classify_complexity(self, query: str, model: str = settings.DEFAULT_MODEL) -> str:
        prompt = (
            f"Analyze this customer support query:\n\"{query}\"\n\n"
            "Classify it as either 'Simple' or 'Complex'.\n"
            "- 'Simple': Basic standard queries (e.g., store hours, standard return timeframe, basic shipping fee, simple greetings).\n"
            "- 'Complex': Multi-part questions, edge cases, complaints requiring nuanced judgment, or policy exceptions."
        )

        try:
            config = types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=RouterDecision,
                temperature=0.0
            )
            response = self._call_gemini_with_retry(model, prompt, config)
            decision = RouterDecision.model_validate_json(response.text)
            return decision.complexity
        except Exception as e:
            print(f"[Gemini Retry Failed - Classify Complexity]: {str(e)}")
            return "Complex"  # Fallback to Complex on failure

    def process_query_workflow(
            self,
            query: str,
            model: str = settings.DEFAULT_MODEL,
            use_rag: bool = False,
            domain: str = "vanilla",
            chat_history: list = None
    ):
        context = ""
        if use_rag:
            context = rag_service.retrieve_context(query, domain=domain)

        # 1. Sentiment & Escalation Analysis on current turn
        analysis_prompt = (
            f"Analyze query: \"{query}\". Classify sentiment (Positive, Neutral, Negative) "
            "and decide if human intervention is required (true/false)."
        )

        analysis_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=QueryAnalysis,
            temperature=0.1
        )
        analysis_response = self._call_gemini_with_retry(model, analysis_prompt, analysis_config)
        analysis = QueryAnalysis.model_validate_json(analysis_response.text)
        sentiment = analysis.sentiment
        needs_human = analysis.human_intervention_required

        # 2. Subtask 3 Multi-Turn Response Generation
        if sentiment == "Positive" and needs_human:
            answer = "Thank you! We are routing your query to a human chat agent."
        elif sentiment == "Negative" and needs_human:
            answer = "We sincerely apologize for your experience. We are routing your issue to an account manager."
        else:
            # Build conversation context string
            history_str = ""
            if chat_history:
                history_str = "Prior Conversation History:\n" + "\n".join(
                    [f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}" for msg in chat_history[-6:]]
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

            gen_config = types.GenerateContentConfig(
                temperature=0.0,
                http_options=types.HttpOptions(timeout=30000)
            )

            res = self._call_gemini_with_retry(model, prompt, gen_config)
            answer = res.text.strip()

        return {
            "sentiment": sentiment,
            "human_intervention_required": needs_human,
            "answer": answer
        }

gemini_service = GeminiService()