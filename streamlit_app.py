import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/v1/customer-query"

st.set_page_config(page_title="Multi-Subtask AI Support Chat", page_icon="🤖")

st.title("🤖 Customer Support AI Assistant")

# Sidebar Controls
st.sidebar.header("Configuration")

# 1. Dev UI Toggle (Controls visibility of backend metadata)
dev_ui = st.sidebar.toggle("Dev UI (Show Backend Details)", value=False)

st.sidebar.markdown("---")

# 2. Mode Selection Toggle
mode = st.sidebar.radio(
    "Application Mode:",
    options=["vanilla", "workplace"],
    format_func=lambda
        x: "Vanilla Mode (ShopEase E-Commerce)" if x == "vanilla" else "Workplace Solution (AdTech Solutions)"
)

# 3. Model Provider Selection
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


# Helper function to render assistant message based on Dev UI toggle
def render_assistant_message(msg_data: dict):
    if dev_ui and "sentiment" in msg_data:
        complexity = msg_data.get("complexity")
        cols = st.columns(4 if complexity else 3)
        cols[0].metric("Subtask 1: Sentiment", msg_data["sentiment"])
        cols[1].metric("Subtask 2: Escalate", "Yes" if msg_data["needs_human"] else "No")
        cols[2].metric("RAG Status", "Active" if msg_data["rag_status"] else "Disabled")
        if complexity:
            cols[3].metric("Router Decision", complexity)

        st.markdown(f"**Subtask 3 Response:**\n{msg_data['content']}")

        caption_text = f"Mode: `{msg_data.get('mode', 'vanilla').upper()}` | Executed via `{msg_data['model_used']}`"
        if msg_data.get("routed_to"):
            caption_text += f" | Routed Target: `{msg_data['routed_to']}`"
        st.caption(caption_text)
    else:
        # Clean End-User View: Show only the response
        st.markdown(msg_data["content"])


# Render Prior Conversation History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            render_assistant_message(message)
        else:
            st.markdown(message["content"])

# Multi-Turn User Follow-up Input
if user_query := st.chat_input("Ask a question or follow up..."):
    formatted_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        spinner_text = "Analyzing subtasks, routing & generating answer..." if dev_ui else "Thinking..."
        with st.spinner(spinner_text):
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
                    timeout=180
                )

                if response.status_code == 200:
                    data = response.json()

                    assistant_msg = {
                        "role": "assistant",
                        "content": data.get("answer", ""),
                        "sentiment": data.get("sentiment", "N/A"),
                        "needs_human": data.get("human_intervention_required", False),
                        "rag_status": data.get("rag_enabled", False),
                        "model_used": data.get("model_used", ""),
                        "complexity": data.get("query_complexity"),
                        "routed_to": data.get("routed_to"),
                        "mode": mode
                    }

                    render_assistant_message(assistant_msg)
                    st.session_state.messages.append(assistant_msg)
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")