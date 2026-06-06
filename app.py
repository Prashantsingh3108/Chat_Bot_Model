import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ---------------- LOAD ENV ----------------

load_dotenv()

# ---------------- STREAMLIT CONFIG ----------------

st.set_page_config(
page_title="RAG Chatbot",
page_icon="🤖",
layout="centered"
)

st.title("🤖 RAG Chatbot")
st.write("Ask questions from your knowledge base.")

# ---------------- API KEY ----------------

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
st.error("❌ GROQ_API_KEY not found.")
st.stop()

# ---------------- LOAD LLM ----------------

@st.cache_resource
def load_llm():
return ChatGroq(
groq_api_key=groq_api_key,
model_name="llama-3.3-70b-versatile",
temperature=0
)

# ---------------- LOAD EMBEDDINGS ----------------

@st.cache_resource
def load_embeddings():
return HuggingFaceEmbeddings(
model_name="sentence-transformers/all-mpnet-base-v2"
)

# ---------------- LOAD VECTOR DB ----------------

@st.cache_resource
def load_vectorstore():

```
embeddings = load_embeddings()

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

return vectorstore
```

# ---------------- INITIALIZATION ----------------

try:
llm = load_llm()
vector_store = load_vectorstore()

except Exception as e:
st.error(f"❌ Failed to load application: {e}")
st.stop()

# ---------------- CHAT HISTORY ----------------

if "messages" not in st.session_state:
st.session_state.messages = []

# Display old messages

for msg in st.session_state.messages:
with st.chat_message(msg["role"]):
st.markdown(msg["content"])

# ---------------- USER INPUT ----------------

user_query = st.chat_input("Ask your question...")

if user_query:

```
# save user message
st.session_state.messages.append(
    {"role": "user", "content": user_query}
)

with st.chat_message("user"):
    st.markdown(user_query)

with st.chat_message("assistant"):

    with st.spinner("Searching knowledge base..."):

        try:

            # similarity search
            docs = vector_store.similarity_search(
                user_query,
                k=5
            )

            # build context
            context = "\n\n".join(
                doc.page_content for doc in docs
            )

            # prompt
            prompt = f"""
```

You are a helpful AI assistant.

Answer ONLY from the provided context.

If answer is not present, say:
"I could not find this information in the knowledge base."

Context:
{context}

Question:
{user_query}
"""

```
            # LLM response
            response = llm.invoke(prompt)

            answer = response.content

            st.markdown(answer)

            # save assistant response
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            # retrieved docs
            with st.expander("Retrieved Context"):
                st.write(context)

        except Exception as e:
            st.error(f"❌ Error: {e}")
```
