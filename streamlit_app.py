import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/v1/customer-query"

st.set_page_config(page_title="Multi-Subtask AI Support Chat", page_icon="🤖")

st.title("🤖 Customer Support AI - Multi-Turn Conversational Assistant")

# Sidebar Controls
st.sidebar.header("Configuration")

mode = st.sidebar.radio(
    "Application Mode:",
    options=["vanilla", "workplace"],
    format_func=lambda x: "Vanilla Mode (ShopEase E-Commerce)" if x == "vanilla" else "Workplace Solution (AdTech Solutions)"
)

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

if st.sidebar.button("Clear Chat Conversation"):
    st.session_state.messages = []
    st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sentiment" in message:
            cols = st.columns(4 if message.get("complexity") else 3)
            cols[0].metric("Subtask 1: Sentiment", message["sentiment"])
            cols[1].metric("Subtask 2: Escalate", "Yes" if message["needs_human"] else "No")
            cols[2].metric("RAG Status", "Active" if message["rag_status"] else "Disabled")
            if message.get("complexity"):
                cols[3].metric("Router Decision", message["complexity"])

            st.markdown(f"**Subtask 3 Response:**\n{message['content']}")
            st.caption(f"Mode: `{message.get('mode', 'vanilla').upper()}` | Executed via `{message['model_used']}`")
        else:
            st.markdown(message["content"])

# Multi-Turn User Follow-up Input
if user_query := st.chat_input("Ask a question or follow up..."):
    # Format current history for payload before appending new query
    formatted_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing query context & generating answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": user_query,
                        "provider": model_provider,
                        "use_rag": use_rag,
                        "mode": mode,
                        "chat_history": formatted_history
                    },
                    timeout=90
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

                    # Append assistant turn to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sentiment": sentiment,
                        "needs_human": needs_human,
                        "rag_status": rag_status,
                        "model_used": model_used,
                        "complexity": complexity,
                        "mode": mode
                    })
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")