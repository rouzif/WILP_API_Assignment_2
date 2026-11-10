import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/v1/customer-query"

st.set_page_config(page_title="Multi-Subtask AI Support", page_icon="🤖")

st.title("🤖 Customer Support AI (3-Subtask Workflow)")

# Sidebar Controls
st.sidebar.header("Configuration")

# Model Provider Radio Selection (4 Options)
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

use_rag = st.sidebar.checkbox("Enable RAG (Policy Knowledge Base)", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Prior Conversation History with Stored Metadata
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sentiment" in message:
            cols = st.columns(4 if message.get("complexity") else 3)
            cols[0].metric("Subtask 1: Sentiment", message["sentiment"])
            cols[1].metric("Subtask 2: Human Escalate", "Yes" if message["needs_human"] else "No")
            cols[2].metric("RAG Status", "Active" if message["rag_status"] else "Disabled")

            if message.get("complexity"):
                cols[3].metric("Router Decision", message["complexity"])

            st.markdown(f"**Subtask 3 Response:**\n{message['content']}")
            caption_text = f"Executed via `{message['model_used']}` | RAG Enabled: `{message['rag_status']}`"
            if message.get("routed_to"):
                caption_text += f" | Routed Target: `{message['routed_to']}`"
            st.caption(caption_text)
        else:
            st.markdown(message["content"])

# Process User Query Input
if user_query := st.chat_input("Enter your customer query..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing query, routing & generating response..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": user_query,
                        "provider": model_provider,
                        "use_rag": use_rag
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
                    routed_to = data.get("routed_to")

                    # Display Metrics (4 columns when router is active)
                    if complexity:
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Subtask 1: Sentiment", sentiment)
                        col2.metric("Subtask 2: Escalate", "Yes" if needs_human else "No")
                        col3.metric("RAG Status", "Active" if rag_status else "Disabled")
                        col4.metric("Router Decision", complexity)
                    else:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Subtask 1: Sentiment", sentiment)
                        col2.metric("Subtask 2: Escalate", "Yes" if needs_human else "No")
                        col3.metric("RAG Status", "Active" if rag_status else "Disabled")

                    st.markdown(f"**Subtask 3 Response:**\n{answer}")

                    caption_text = f"Executed via `{model_used}` | RAG Enabled: `{rag_status}`"
                    if routed_to:
                        caption_text += f" | Routed Target: `{routed_to}`"
                    st.caption(caption_text)

                    # Store full metadata into session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sentiment": sentiment,
                        "needs_human": needs_human,
                        "rag_status": rag_status,
                        "model_used": model_used,
                        "complexity": complexity,
                        "routed_to": routed_to
                    })
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")