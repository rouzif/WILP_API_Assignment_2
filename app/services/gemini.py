import json
from google import genai
from google.genai import types
from app.config import settings
from app.schemas import QueryAnalysis
from app.services.rag import rag_service


class GeminiService:
    def __init__(self):
        self.client = genai.Client(api_key=settings.GEMINI_API_KEY)

    def process_query_workflow(self, query: str, model: str = settings.DEFAULT_MODEL, use_rag: bool = False):

        context = ""
        if use_rag:
            context = rag_service.retrieve_context(query)
            print(f"[DEBUG RAG CONTEXT]:\n{context}")  # Check your console to confirm context is retrieved!

        # 1. Perform Subtask 1 & 2 using Structured Outputs
        analysis_prompt = (
            f"Analyze the following customer query:\n\n\"{query}\"\n\n"
            "1. Classify sentiment as Positive, Neutral, or Negative.\n"
            "2. Determine if human intervention is required (True/False)."
        )

        analysis_response = self.client.models.generate_content(
            model=model,
            contents=analysis_prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=QueryAnalysis,
                temperature=0.1
            )
        )

        analysis = QueryAnalysis.model_validate_json(analysis_response.text)
        sentiment = analysis.sentiment
        needs_human = analysis.human_intervention_required

        # 3. Subtask 3 Logic & Response Generation
        if sentiment == "Positive" and needs_human:
            answer = "Thank you for reaching out! We appreciate your positive feedback. We are routing your query to a human chat agent right away."

        elif sentiment == "Neutral" and needs_human:
            answer = "We have received your request and we are routing your request directly to an account manager."

        elif sentiment == "Negative" and needs_human:
            answer = "We sincerely apologize for the experience you have faced. Your issue is being prioritized, and we are routing your request directly to an account manager."

        elif sentiment == "Negative" and not needs_human:
            gen_prompt = f"The customer query is negative: '{query}'. Apologize to the customer and respond to address their concern."
            gen_prompt += f"\nUse the following ShopEase Official Policy Context to answer:\n{context}\n"
            res = self.client.models.generate_content(model=model, contents=gen_prompt)
            answer = res.text.strip()


        else:

            if use_rag and context:
                # STRICT GROUNDING PROMPT FOR GEMINI
                prompt = (
                    "You are a customer support agent for ShopEase.\n"
                    "STRICT INSTRUCTIONS: Answer the customer's question strictly using ONLY the provided policy context below. "
                    "Do NOT invent facts, placeholders (like [Number]), or assume standard industry terms. "
                    "If the answer is not contained in the context, say 'I do not have information on that policy.'\n\n"
                    f"Official Policy Context:\n{context}\n\n"
                    f"Customer Question: {query}"
                )

            else:
                prompt = f"Customer Query: {query}\nProvide a general customer support answer."

            res = self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0)  # Lower temperature for strictly factual answers
            )

            answer = res.text.strip()

        return {
            "sentiment": sentiment,
            "human_intervention_required": needs_human,
            "answer": answer
        }


gemini_service = GeminiService()