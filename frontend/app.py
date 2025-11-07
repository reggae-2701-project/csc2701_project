import streamlit as st
import requests

API_URL = "http://172.31.5.225:8080/chat"
st.set_page_config(page_title="MScAC Chatbot", page_icon="💬", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center;'> MScAC Information Chatbot</h1>
    <p style='text-align: center;'>Ask me anything about the program!</p>
    """,
    unsafe_allow_html=True
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")

        try:
            response = requests.post(API_URL, json={"messages": st.session_state.messages})
            response.raise_for_status()
            data = response.json()
            answer = data.get("response", "No response received.")
            placeholder.markdown(answer)

        except Exception as e:
            answer = f"Error: {e}"
            placeholder.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
