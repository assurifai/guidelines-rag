import openai
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer

from ingest import Embs, get_table

# Load the table for embedding search
tbl = get_table()

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("📄 RAG Chatbot with Streamlit")
# Sidebar controls
st.sidebar.header("Settings")
num_contexts = st.sidebar.slider("Number of contexts to retrieve", 1, 10, 3)
if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "page_to_render" not in st.session_state:
    st.session_state.page_to_render = 1

chat, pdf = st.columns([1, 1])
with chat:
    with st.container(height=700):
        # Chat Input
        user_input = st.chat_input("Ask me anything...")
        for chat_item in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat_item["user"])
            with st.chat_message("assistant"):
                st.write(chat_item["bot"])
            with st.chat_message("assistant"):
                with st.expander("Show Sources"):
                    unique_pages = chat_item["sources"]
                    for i in range(0, len(unique_pages), 5):
                        cols = st.columns(5)
                        for j, col in enumerate(cols):
                            if i + j < len(unique_pages):
                                with col:
                                    if st.button(
                                        f"Page {unique_pages[i + j]}", type="tertiary"
                                    ):
                                        st.session_state.page_to_render = unique_pages[
                                            i + j
                                        ]

        if user_input:
            with st.chat_message("user"):
                response = st.write(user_input)
            # Perform retrieval
            results = tbl.search(user_input).limit(num_contexts).to_pydantic(Embs)
            retrieved_texts = [result.text for result in results]
            retrieved_pages = [result.pg_numb for result in results]
            st.session_state.page_to_render = max(
                set(retrieved_pages), key=retrieved_pages.count
            )  # Mode of the list
            # # Display retrieved chunks
            # with st.expander("Retrieved Documents"):
            #     for i, text in enumerate(retrieved_texts):
            #         st.markdown(f"**Page {retrieved_pages[i]}:** {text}")

            # Generate response using OpenAI
            messages = [
                {
                    "role": "system",
                    "content": "You are a RAG Chatbot for a lender. The relevant context is pulled from Fannie Mae guidelines, answer the questions based on Fannie Mae's guidelines.",
                },
                {"role": "user", "content": user_input},
                {
                    "role": "user",
                    "content": "Relevant context: " + "\n\n".join(retrieved_texts),
                },
            ]
            client = openai.OpenAI()  # Create an OpenAI client

            response_generator = client.chat.completions.create(
                model="gpt-4o-mini", messages=messages, stream=True  # Enable streaming
            )
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                response = st.write_stream(response_generator)
            with st.chat_message("assistant"):
                with st.expander("Show Sources"):
                    unique_pages = list(set(retrieved_pages))
                    for i in range(0, len(unique_pages), 5):
                        cols = st.columns(5)
                        for j, col in enumerate(cols):
                            if i + j < len(unique_pages):
                                with col:
                                    if st.button(
                                        f"Page {unique_pages[i + j]}", type="tertiary"
                                    ):
                                        st.session_state.page_to_render = unique_pages[
                                            i + j
                                        ]

            # Update chat history
            st.session_state.chat_history.append(
                {"user": user_input, "bot": response, "sources": unique_pages}
            )

with pdf:
    pdf_viewer(
        "Selling-Guide_02-05-25_highlighted.pdf",
        scroll_to_page=st.session_state.page_to_render,
        height=700,
    )
