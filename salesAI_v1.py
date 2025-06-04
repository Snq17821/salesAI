import streamlit as st
import os
from pathlib import Path
import time

# LlamaIndex core imports
from llama_index.core import VectorStoreIndex, Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore

# Pinecone client library
from pinecone import Pinecone

# --- Configuration ---
# Set your API keys and environment variables.
# Ensure these are set in your environment or replace "YOUR_..." placeholders.
# For Streamlit Cloud, you'd typically set these as secrets.
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
# PINECONE_ENVIRONMENT is often required, but some Pinecone setups (e.g., serverless in a default region)
# might infer it or not strictly require it in the constructor. Uncomment and set if needed.
# PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "YOUR_PINECONE_ENVIRONMENT") # e.g., "us-west-2"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Define your Pinecone index name (must match the one used for ingestion)
INDEX_NAME = "playbook" # Ensure this matches your ingested index name

# --- LlamaIndex Global Settings Configuration ---
# Configure the Language Model (LLM) for generating responses.
# 'gpt-3.5-turbo' is a common choice, but you can use 'gpt-4o' or others.
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1, api_key=OPENAI_API_KEY)

# Configure the Embedding Model for querying the vector database.
# This must match the model used for ingesting data (text-embedding-3-large).
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large", api_key=OPENAI_API_KEY)

# --- Streamlit App ---
st.set_page_config(
    page_title="Unipal Sales GPT", 
    layout="centered", # Can also be "wide"
    initial_sidebar_state="collapsed" # "auto", "expanded", "collapsed"
)

# Inject custom CSS for font and general styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #0e1117; /* Dark background similar to original */
        color: #fafafa;
    }
    .st-emotion-cache-z5fcl4 { /* Target the main block container for padding */
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }
    .st-emotion-cache-1c7y2kl { /* Target the chat input container */
        padding-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📚 Unipal Sales GPT")
st.markdown("Ask any question about sales!")
st.divider() # Adds a subtle horizontal line

# Initialize LlamaIndex VectorStoreIndex (cached to avoid re-loading on every rerun)
@st.cache_resource(show_spinner=False)
def get_index():
    with st.spinner(
        "Connecting to Pinecone and loading your knowledge base... This might take a moment!"
    ):
        # Initialize Pinecone client
        try:
            # Initialize Pinecone without environment if it's not needed for your setup
            # Otherwise, use: pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            pc = Pinecone(api_key=PINECONE_API_KEY)
        except Exception as e:
            st.error(f"Error connecting to Pinecone: {e}. Please check your API key.")
            st.stop() # Stop the app if connection fails

        # Connect to the specific Pinecone index
        try:
            pinecone_index = pc.Index(INDEX_NAME)
        except Exception as e:
            st.error(f"Error connecting to index '{INDEX_NAME}': {e}. Ensure the index exists and is active.")
            st.stop()

        # Create a LlamaIndex PineconeVectorStore instance
        vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

        # Re-initialize the VectorStoreIndex from the existing vector store
        # This step sets up the query engine correctly, as the data is already in Pinecone.
        try:
            index = VectorStoreIndex.from_vector_store(vector_store)
            return index
        except Exception as e:
            st.error(f"Error initializing LlamaIndex from vector store: {e}. Ensure data was successfully indexed.")
            st.stop()

# Get the index (will be cached after first run)
index = get_index()

# Create a query engine from the index
# This engine will retrieve relevant chunks from Pinecone and use the LLM to synthesize an answer.
query_engine = index.as_query_engine()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about the playbook..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from the query engine
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        with st.spinner("Thinking..."):
            try:
                response = query_engine.query(prompt)
                full_response = response.response # Access the response string
            except Exception as e:
                full_response = f"An error occurred: {e}. Please check your OpenAI API key and ensure the index is properly configured."
            
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
