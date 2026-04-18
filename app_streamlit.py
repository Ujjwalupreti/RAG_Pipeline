import streamlit as st
import os
from langchain_core.messages import HumanMessage, AIMessage
from src.generation import get_conversational_rag_chain
from src.ingestion import ingest_documents

st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Advanced Enterprise RAG System")
st.markdown("Powered by Hybrid Search, Cross-Encoder Re-ranking, and Gemini.")

@st.cache_resource
def initialize_system():
    if not os.path.exists("./vector_store") or not os.listdir("./vector_store"):
        with st.spinner("First-time setup: Ingesting documents. This may take a moment..."):
            ingest_documents()
    return get_conversational_rag_chain()

rag_chain = initialize_system()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

if prompt := st.chat_input("Ask a question about your documents..."):
    
    st.session_state.chat_history.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & analyzing context..."):

            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history[:-1]
            })
            
            answer = response["answer"]
            context_docs = response["context"]

            st.markdown(answer)
            
            with st.expander("📄 View Source Documents Used"):
                if not context_docs:
                    st.warning("No relevant documents found.")
                else:
                    for i, doc in enumerate(context_docs):
                        page_num = doc.metadata.get('page', 'Unknown')
                        source_file = os.path.basename(doc.metadata.get('source', 'Unknown File'))
                        
                        st.markdown(f"**Source {i+1} | File:** `{source_file}` **| Page:** `{page_num}`")
                        st.info(doc.page_content)

    st.session_state.chat_history.append(AIMessage(content=answer))