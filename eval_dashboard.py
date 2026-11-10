import streamlit as st
import json
import os

st.set_page_config(page_title="AdTech AI Benchmark Dashboard", page_icon="📊", layout="wide")

st.title("📊 AdTech Workplace AI Solutions - Benchmark Evaluation")

RESULTS_PATH = "./data/eval_results.json"

if not os.path.exists(RESULTS_PATH):
    st.warning("No evaluation results found. Run `python scripts/evaluate_pipeline.py` first.")
else:
    with open(RESULTS_PATH, "r") as f:
        data = json.load(f)

    # Summary Metrics Header (Including Correctness Score)
    st.subheader("1. Overall Performance Statistics")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Total Queries", data["total_queries"])
    c2.metric("Sentiment Acc.", f"{data['sentiment_accuracy']}%")
    c3.metric("Escalation Acc.", f"{data['escalation_accuracy']}%")
    c4.metric("Router Acc.", f"{data['complexity_accuracy']}%")
    c5.metric("Grounding Rate", f"{data['grounding_accuracy']}%")
    c6.metric("Mean Correctness", f"{data.get('mean_correctness_score', 'N/A')}/10")
    c7.metric("Avg Latency", f"{data['avg_latency_sec']}s")

    st.markdown("---")

    # Itemized Breakdown
    st.subheader("2. Detailed Query Inspection & Judge Rubric Scores")
    for item in data["details"]:
        score = item.get("correctness_score", "N/A")
        with st.expander(f"Query {item['id']}: {item['query']} — Score: {score}/10"):
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.write(f"**Sentiment:** {item['predicted_sentiment']} (Ideal: {item['ideal_sentiment']})")
            col_b.write(f"**Escalation:** {item['predicted_escalation']} (Ideal: {item['ideal_escalation']})")
            col_c.write(f"**Complexity:** {item['predicted_complexity']} (Ideal: {item['ideal_complexity']})")
            col_d.write(f"**Grounded:** {'✅ Yes' if item['grounded'] else '❌ No'}")

            st.write("**Ideal Answer:**")
            st.success(item.get("ideal_answer", "N/A"))

            st.write("**Generated Response:**")
            st.info(item["generated_answer"])

            st.write("**LLM Judge Reasoning:**")
            st.caption(item.get("correctness_reasoning", "N/A"))