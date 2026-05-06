import os
import streamlit as st
from pipeline.rag import query
from clients.vecdb_client import VectorDBClient
from clients.llm_client import LLMClient

vecdb_client = VectorDBClient()
llm_client = LLMClient(os.environ["GROQ_API_KEY"])

st.title("ArXiv Research Assistant")
question = st.text_input("Ask about recent AI research:")
if question:
    with st.spinner("Searching..."):
        answer = query(question, vecdb_client, llm_client)
    st.write(answer)