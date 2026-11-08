import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/v1/customer-query"

st.set_page_config(page_title="Multi-Model Support AI", page_icon="🤖")

st.title("🤖 Customer Support AI")

# Sidebar model selector featuring Base vs Fine-Tuned vs Cloud API
st.sidebar.header("Model Selection")
model_provider = st.sidebar.radio(
    "Choose Active Model:",
    options=["gemini", "base", "finetuned"],
    format_func=lambda x: {
        "gemini": "Gemini 2.5 Flash Lite (Cloud API)",
        "base": "Gemma 2B (Base Un-tuned SLM)",
        "finetuned": "Gemma 2B (Fine-Tuned SLM)"
    }[x]
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input("Ask a customer support question..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                response = requests.post(
                    API_URL,
                    json={
                        "query": user_query,
                        "provider": model_provider
                    },
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer")
                    model_used = data.get("model_used")
                    provider_used = data.get("provider")

                    st.markdown(answer)
                    st.caption(f"Provider: `{provider_used}` | Model: `{model_used}`")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {str(e)}")