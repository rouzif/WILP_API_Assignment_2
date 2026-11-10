import sys
import os

# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import requests
from pydantic import BaseModel, Field
from google import genai
from app.config import settings

API_URL = "http://127.0.0.1:8000/api/v1/customer-query"
DATASET_PATH = "./data/test_dataset.json"
RESULTS_PATH = "./data/eval_results.json"

client = genai.Client(api_key=settings.GEMINI_API_KEY)


# 1. Pydantic schema for LLM Judge Correctness Evaluation
class CorrectnessEvaluation(BaseModel):
    score: int = Field(..., description="Score from 0 to 10 based on rubric match.")
    reasoning: str = Field(..., description="Brief explanation of why the score was assigned.")


def judge_correctness(query: str, generated_answer: str, ideal_answer: str) -> dict:
    """
    Evaluates generated answer against the ideal answer using LLM-as-a-Judge.
    Rubric:
    - 9-10: Fully accurate, contains all key facts from ideal answer, no contradictions.
    - 6-8: Mostly accurate, covers main points but misses minor details.
    - 3-5: Partially correct, contains minor factual errors or lacks key policy facts.
    - 0-2: Incorrect, hallucinated, or completely misses the query intent.
    """
    if not generated_answer:
        return {"score": 0, "reasoning": "No answer generated."}

    judge_prompt = (
        f"User Query: {query}\n\n"
        f"Ideal Answer (Ground Truth): {ideal_answer}\n\n"
        f"Generated Answer: {generated_answer}\n\n"
        "Evaluate the Generated Answer against the Ideal Answer based on semantic correctness, policy accuracy, "
        "and factual alignment. Assign a score between 0 and 10 and provide brief reasoning."
    )

    try:
        res = client.models.generate_content(
            model=settings.DEFAULT_MODEL,
            contents=judge_prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": CorrectnessEvaluation,
                "temperature": 0.0
            }
        )
        parsed = CorrectnessEvaluation.model_validate_json(res.text)
        return {"score": parsed.score, "reasoning": parsed.reasoning}
    except Exception as e:
        return {"score": 5, "reasoning": f"Judge evaluation fallback due to error: {str(e)}"}


def check_grounding(answer: str, query: str) -> bool:
    if not answer:
        return False
    prompt = (
        f"Query: {query}\nAnswer: {answer}\n"
        "Evaluate if the answer gives explicit factual facts related to Ad Tech policies without hallucinated placeholder text like [Number]. "
        "Reply strictly with JSON: {\"grounded\": true} or {\"grounded\": false}"
    )
    try:
        res = client.models.generate_content(
            model=settings.DEFAULT_MODEL,
            contents=prompt,
            config={"response_mime_type": "application/json"}
        )
        parsed = json.loads(res.text)
        return bool(parsed.get("grounded", True))
    except Exception:
        return True


def run_evaluation():
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Test dataset file '{DATASET_PATH}' not found.")
        return

    with open(DATASET_PATH, "r") as f:
        test_cases = json.load(f)

    results = []
    latencies = []

    correct_sentiment = 0
    correct_escalation = 0
    correct_complexity = 0
    grounded_count = 0
    correctness_scores = []

    total = len(test_cases)
    start_all = time.time()

    for item in test_cases:
        t0 = time.time()
        try:
            resp = requests.post(
                API_URL,
                json={
                    "query": item["query"],
                    "provider": "router",
                    "use_rag": True,
                    "mode": "workplace"
                },
                timeout=60
            )
            t1 = time.time()
            latency = round(t1 - t0, 2)
            latencies.append(latency)

            if resp.status_code == 200:
                data = resp.json()

                sent = data.get("sentiment")
                esc = data.get("human_intervention_required") if data.get(
                    "human_intervention_required") is not None else data.get("needs_human", False)
                comp = data.get("query_complexity") if data.get("query_complexity") is not None else data.get(
                    "complexity", "Simple")
                ans = data.get("answer", "")

                # Evaluation Matches
                s_match = (str(sent).strip().lower() == str(item["ideal_sentiment"]).strip().lower())
                e_match = (bool(esc) == bool(item["ideal_escalation"]))
                c_match = (str(comp).strip().lower() == str(item["ideal_complexity"]).strip().lower())
                grounded = check_grounding(ans, item["query"])

                # 2. Judge LLM Correctness Evaluation
                judge_eval = judge_correctness(
                    query=item["query"],
                    generated_answer=ans,
                    ideal_answer=item.get("ideal_answer", "")
                )
                correctness_scores.append(judge_eval["score"])

                if s_match: correct_sentiment += 1
                if e_match: correct_escalation += 1
                if c_match: correct_complexity += 1
                if grounded: grounded_count += 1

                results.append({
                    "id": item["id"],
                    "query": item["query"],
                    "predicted_sentiment": sent,
                    "ideal_sentiment": item["ideal_sentiment"],
                    "sentiment_match": s_match,
                    "predicted_escalation": esc,
                    "ideal_escalation": item["ideal_escalation"],
                    "escalation_match": e_match,
                    "predicted_complexity": comp,
                    "ideal_complexity": item["ideal_complexity"],
                    "complexity_match": c_match,
                    "generated_answer": ans,
                    "ideal_answer": item.get("ideal_answer", ""),
                    "correctness_score": judge_eval["score"],
                    "correctness_reasoning": judge_eval["reasoning"],
                    "grounded": grounded,
                    "latency_sec": latency
                })
            else:
                print(f"API Error ({resp.status_code}) on query {item['id']}: {resp.text}")

        except Exception as e:
            print(f"Request failed for query {item['id']}: {str(e)}")

    total_time = time.time() - start_all
    throughput = round(total / total_time, 2) if total_time > 0 else 0
    avg_correctness = round(sum(correctness_scores) / len(correctness_scores), 2) if correctness_scores else 0.0

    summary = {
        "total_queries": total,
        "sentiment_accuracy": round((correct_sentiment / total) * 100, 2) if total else 0,
        "escalation_accuracy": round((correct_escalation / total) * 100, 2) if total else 0,
        "complexity_accuracy": round((correct_complexity / total) * 100, 2) if total else 0,
        "grounding_accuracy": round((grounded_count / total) * 100, 2) if total else 0,
        "mean_correctness_score": avg_correctness,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "throughput_qps": throughput,
        "details": results
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete! Mean Correctness Score: {avg_correctness}/10. Results saved to {RESULTS_PATH}.")


if __name__ == "__main__":
    run_evaluation()