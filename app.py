import openai
import streamlit as st

from ingest import Embs, get_table

# Load the table for embedding search
tbl = get_table()

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot with Streamlit")

# Sidebar for OpenAI API key
# openai_api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat Input
user_input = st.chat_input("Ask me anything...")

if user_input:
    # Perform retrieval
    results = tbl.search(user_input).limit(10).to_pydantic(Embs)
    retrieved_texts = [result.text for result in results]
    retrieved_pages = [result.pg_numb for result in results]

    # Display retrieved chunks
    with st.expander("Retrieved Documents"):
        for i, text in enumerate(retrieved_texts):
            st.markdown(f"**Page {retrieved_pages[i]}:** {text}")

    # Generate response using OpenAI
    messages = [
        {
            "role": "system",
            "content": "You are a RAG Chatbot for a lender. The relevant context is pulled from fannie mae guidelines, answer the questions based on fannie mae's guidelines.",
        }
    ]
    messages.append({"role": "user", "content": user_input})
    messages.append(
        {
            "role": "user",
            "content": "Relevant context: " + "\n\n".join(retrieved_texts),
        }
    )

    client = openai.OpenAI()  # Create an OpenAI client

    response = client.chat.completions.create(model="gpt-4o", messages=messages)

    assistant_reply = response.choices[0].message.content

    # Update chat history
    st.session_state.chat_history.append({"user": user_input, "bot": assistant_reply})

    # Display chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["user"])
        with st.chat_message("assistant"):
            st.markdown(chat["bot"])
