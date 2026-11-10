import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/v1/customer-query"

st.set_page_config(page_title="Multi-Subtask AI Support", page_icon="🤖")

st.title("🤖 Customer Support AI Assistant")

# Sidebar Controls
st.sidebar.header("Configuration")

# 1. Mode Selection Toggle
mode = st.sidebar.radio(
    "Application Mode:",
    options=["vanilla", "workplace"],
    format_func=lambda x: "Vanilla Mode (ShopEase E-Commerce)" if x == "vanilla" else "Workplace Solution (AdTech Solutions)"
)

# 2. Model Provider Selection
model_provider = st.sidebar.radio(
    "Choose Active Model:",
    options=["gemini", "base", "finetuned", "router"],
    format_func=lambda x: {
        "gemini": "Gemini 2.5 Flash Lite (Cloud API)",
        "base": "Gemma 2B (Base SLM)",
        "finetuned": "Gemma 2B (Fine-Tuned SLM)",
        "router": "Dynamic Router (Gemini Classifier -> Gemma / Gemini)"
    }[x]
)

use_rag = st.sidebar.checkbox("Enable RAG Knowledge Base", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Enter your customer support query..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Processing query..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": user_query,
                        "provider": model_provider,
                        "use_rag": use_rag,
                        "mode": mode
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()

                    sentiment = data.get("sentiment", "N/A")
                    needs_human = data.get("human_intervention_required", False)
                    answer = data.get("answer", "")
                    model_used = data.get("model_used", "")
                    rag_status = data.get("rag_enabled", False)
                    complexity = data.get("query_complexity")

                    cols = st.columns(4 if complexity else 3)
                    cols[0].metric("Subtask 1: Sentiment", sentiment)
                    cols[1].metric("Subtask 2: Escalate", "Yes" if needs_human else "No")
                    cols[2].metric("RAG Status", "Active" if rag_status else "Disabled")
                    if complexity:
                        cols[3].metric("Router Decision", complexity)

                    st.markdown(f"**Subtask 3 Response:**\n{answer}")
                    st.caption(f"Mode: `{mode.upper()}` | Executed via `{model_used}`")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")