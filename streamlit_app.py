import streamlit as st
import requests

# FastAPI Backend Endpoint Configuration
API_URL = "http://127.0.0.1:8000/api/v1/customer-query"

# Streamlit Page Setup
st.set_page_config(
    page_title="AI Customer Support Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Customer Support Assistant")
st.markdown("Ask any question regarding our products, orders, or support policies.")

# Initialize chat session history if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input Interface
if user_query := st.chat_input("How can we help you today?"):
    # Render user query in chat
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Call FastAPI backend and display response
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": user_query},
                    headers={"Content-Type": "application/json"},
                    timeout=30
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data.get("answer", "No response content received.")
                    model_used = data.get("model_used", "Gemini")

                    st.markdown(answer)
                    st.caption(f"Powered by `{model_used}`")

                    # Store assistant response in history
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    error_msg = f"API Error ({response.status_code}): {response.text}"
                    st.error(error_msg)

            except requests.exceptions.ConnectionError:
                st.error(
                    "Could not connect to the API backend. Make sure the FastAPI server is running on http://127.0.0.1:8000.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")