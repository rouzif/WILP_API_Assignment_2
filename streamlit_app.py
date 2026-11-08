import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/v1/customer-query"

st.set_page_config(page_title="Multi-Subtask AI Support", page_icon="🤖")

st.title("🤖 Customer Support AI (3-Subtask Workflow)")

# Sidebar Controls
st.sidebar.header("Configuration")

model_provider = st.sidebar.radio(
    "Choose Active Model:",
    options=["gemini", "base", "finetuned"],
    format_func=lambda x: {
        "gemini": "Gemini 2.5 Flash Lite (Cloud API)",
        "base": "Gemma 2B (Base SLM)",
        "finetuned": "Gemma 2B (Fine-Tuned SLM)"
    }[x]
)

use_rag = st.sidebar.checkbox("Enable RAG (Policy Knowledge Base)", value=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Prior Conversation History with Stored Metadata
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and "sentiment" in message:
            col1, col2, col3 = st.columns(3)
            col1.metric("Subtask 1: Sentiment", message["sentiment"])
            col2.metric("Subtask 2: Human Escalate", "Yes" if message["needs_human"] else "No")
            col3.metric("RAG Status", "Active" if message["rag_status"] else "Disabled")

            st.markdown(f"**Subtask 3 Response:**\n{message['content']}")
            st.caption(f"Executed via `{message['model_used']}` | RAG Enabled: `{message['rag_status']}`")
        else:
            st.markdown(message["content"])

# Process User Query Input
if user_query := st.chat_input("Enter your customer query..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing subtasks & generating response..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": user_query,
                        "provider": model_provider,
                        "use_rag": use_rag
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()

                    sentiment = data.get("sentiment", "N/A")
                    needs_human = data.get("human_intervention_required", False)
                    answer = data.get("answer", "")
                    model_used = data.get("model_used", "")
                    rag_status = data.get("rag_enabled", False)

                    # Render for current response turn
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Subtask 1: Sentiment", sentiment)
                    col2.metric("Subtask 2: Human Escalate", "Yes" if needs_human else "No")
                    col3.metric("RAG Status", "Active" if rag_status else "Disabled")

                    st.markdown(f"**Subtask 3 Response:**\n{answer}")
                    st.caption(f"Executed via `{model_used}` | RAG Enabled: `{rag_status}`")

                    # Store full metadata into session state
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sentiment": sentiment,
                        "needs_human": needs_human,
                        "rag_status": rag_status,
                        "model_used": model_used
                    })
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")