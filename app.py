import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

# ---------------- PAGE CONFIG ---------------- #

st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 RAG Chatbot")
st.write("Ask questions from your uploaded knowledge base.")

# ---------------- API KEY CHECK ---------------- #

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables.")
    st.stop()

# ---------------- LOAD MODEL ---------------- #

@st.cache_resource
def load_llm():
    return ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

# ---------------- LOAD EMBEDDINGS ---------------- #

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

# ---------------- LOAD VECTOR DATABASE ---------------- #

@st.cache_resource
def load_vectorstore():

    embeddings = load_embeddings()

    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    return vectorstore

# ---------------- INITIALIZE ---------------- #

try:
    llm = load_llm()
    vector_store = load_vectorstore()

except Exception as e:
    st.error(f"❌ Error loading app: {e}")
    st.stop()

# ---------------- USER INPUT ---------------- #

query = st.text_input("Enter your question:")

if st.button("Submit"):

    if query.strip() == "":
        st.warning("Please enter a question.")
    
    else:

        with st.spinner("Searching documents..."):

            try:

                # similarity search
                results = vector_store.similarity_search(query, k=5)

                # create context
                context = "\n\n".join(
                    doc.page_content for doc in results
                )

                # prompt
                prompt = f"""
You are a helpful AI assistant.

Answer the question using ONLY the context below.

Context:
{context}

Question:
{query}
"""

                # generate response
                response = llm.invoke(prompt)

                # display answer
                st.subheader("Answer")
                st.write(response.content)

                # optional retrieved chunks
                with st.expander("Retrieved Context"):
                    st.write(context)

            except Exception as e:
                st.error(f"❌ Error: {e}")