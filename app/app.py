import os
import streamlit as st
import yaml
import config.paths as paths
from pipeline.rag import query
from clients.vecdb_client import VectorDBClient
from clients.llm_client import LLMClient

with open(paths.MODEL_CONFIG_DIR, "r") as file:
    config = yaml.safe_load(file)

st.title("ArXiv Research Assistant")
st.sidebar.header("Model Configuration")

provider_options = list(config['providers'].keys())
selected_provider = st.sidebar.selectbox("Select Provider", options=provider_options)

if selected_provider is None:
    st.sidebar.warning("Please select a provider to proceed.")
    st.stop() 

model_options = config['providers'][selected_provider]['models']
selected_model = st.sidebar.selectbox("Select Model", options=model_options)

if selected_model is None:
    st.sidebar.warning("Please select a model to proceed.")
    st.stop()

base_url = config['providers'][selected_provider]['base_url']

env_key_name = f"{selected_provider.upper()}_API_KEY"

api_key = os.getenv(env_key_name)

if not api_key:
    st.error(f"Missing API Key: Please ensure `{env_key_name}` is set in your environment.")
    st.stop()

vecdb_client = VectorDBClient()
llm_client = LLMClient(
    provider=selected_provider, 
    api_key=api_key,
    base_url=base_url
)

question = st.text_input("Ask about recent AI research:")

if question:
    with st.spinner("Searching..."):
        answer = query(question, vecdb_client, llm_client, selected_model)
    st.write(answer)