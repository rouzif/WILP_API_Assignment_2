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

    # Summary Metrics Header
    st.subheader("1. Overall Performance Statistics")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Queries", data["total_queries"])
    c2.metric("Sentiment Accuracy", f"{data['sentiment_accuracy']}%")
    c3.metric("Escalation Accuracy", f"{data['escalation_accuracy']}%")
    c4.metric("Router Complexity Acc.", f"{data['complexity_accuracy']}%")
    c5.metric("Policy Grounding Rate", f"{data['grounding_accuracy']}%")
    c6.metric("Avg Latency", f"{data['avg_latency_sec']}s")

    st.markdown("---")

    # Itemized Breakdown Table / Inspection
    st.subheader("2. Detailed Query Inspection")
    for item in data["details"]:
        with st.expander(f"Query {item['id']}: {item['query']}"):
            col_a, col_b, col_c, col_d = st.columns(4)
            col_a.write(f"**Sentiment:** {item['predicted_sentiment']} (Ideal: {item['ideal_sentiment']})")
            col_b.write(f"**Escalation:** {item['predicted_escalation']} (Ideal: {item['ideal_escalation']})")
            col_c.write(f"**Complexity:** {item['predicted_complexity']} (Ideal: {item['ideal_complexity']})")
            col_d.write(f"**Grounded:** {'✅ Yes' if item['grounded'] else '❌ No'}")

            st.write("**Generated Response:**")
            st.info(item["generated_answer"])
            st.caption(f"Latency: {item['latency_sec']} seconds")