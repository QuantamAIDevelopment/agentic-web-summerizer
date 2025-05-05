# main.py
"""
Agentic Web Summarizer - A Streamlit application that uses AI to summarize web content.

This application demonstrates the concept of AI agents by:
1. Fetching content from a web page
2. Using an LLM to generate a concise summary
3. Storing the summary in a vector database for potential future retrieval

The app showcases how to combine different AI capabilities (LLMs, embeddings)
with web scraping to create a useful tool.
"""

import os
import streamlit as st
import logging
from datetime import datetime
import warnings

# Suppress PyTorch warnings that might interfere with Streamlit
warnings.filterwarnings("ignore", category=RuntimeWarning, module="torch._classes")
warnings.filterwarnings("ignore", message=".*Tried to instantiate class.*")

# Fix for asyncio error in Streamlit
import asyncio
try:
    asyncio.get_running_loop()
except RuntimeError:
    # If there's no running event loop, create one
    asyncio.set_event_loop(asyncio.new_event_loop())

# Import our custom modules
from llm.llm_provider import get_llm
from tools.fetcher import fetch_url_content
from tools.summarizer import summarize_text
from tools.vectorstore import get_embeddings, create_faiss_store, store_summary, save_vectorstore

# Import default keys configuration
from default_keys import USE_DEFAULT_KEYS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Agentic Web Summarizer",
    page_icon="🔍",
    layout="wide"
)

# Application title and description
st.title("🔍 Agentic Web Summarizer")
st.markdown("""
This tool demonstrates how AI agents can help process web content:
1. Enter a URL to fetch the content
2. The AI will generate a concise summary
3. The summary is stored in a vector database for future reference
""")

# Sidebar for configuration options
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Keys section
    with st.expander("API Keys", expanded=False):
        if USE_DEFAULT_KEYS:
            st.success("Default API keys are available. You can use your own keys or leave fields empty to use default keys.")
        else:
            st.info("API keys are stored in session state and not saved to disk")
        
        # Use default keys option
        use_default = st.checkbox(
            "Use default API keys when available", 
            value=USE_DEFAULT_KEYS,
            help="When checked, the application will use built-in API keys if you don't provide your own"
        )
        st.info(" If you are not selecting the default keys Choose any one LLM model and pass the API keys to It.")
        # OpenAI API Key
        openai_api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
            help="Your OpenAI API key for accessing GPT models"
        )
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        elif not use_default:
            st.warning("No OpenAI API key provided. Required for OpenAI models and embeddings.")
            
        # Groq API Key
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Your Groq API key for accessing Groq models"
        )
        if groq_api_key:
            os.environ["GROQ_API_KEY"] = groq_api_key
        elif not use_default:
            st.warning("No Groq API key provided. Required for Groq models.")
            
        # LangChain API Key for tracing
        langchain_api_key = st.text_input(
            "LangChain API Key",
            type="password",
            value=os.getenv("LANGCHAIN_API_KEY", ""),
            help="Your LangChain API key for tracing and monitoring"
        )
        if langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = langchain_api_key
    
    # LLM provider selection
    provider = st.selectbox(
        "Choose LLM Provider",
        options=["openai", "groq"],
        help="Select which AI model provider to use for summarization"
    )
    
    # Embedding model selection
    embedder = st.selectbox(
        "Choose Embedding Model",
        options=["openai", "huggingface"],
        help="Select which embedding model to use for vector storage"
    )
    
    # Summary style selection
    summary_style = st.radio(
        "Summary Style",
        options=["default", "bullet"],
        help="Choose how the summary should be formatted"
    )
    
    # Advanced options in an expandable section
    with st.expander("Advanced Options"):
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        max_chars = st.number_input(
            "Max Input Characters",
            min_value=1000,
            max_value=10000,
            value=4000,
            step=1000,
            help="Maximum number of characters to process from the webpage"
        )
        
        save_to_disk = st.checkbox(
            "Save Summaries to Disk",
            value=False,
            help="Store summaries in a local vector database for future reference"
        )
        
        # Model options based on provider
        if provider == "groq":
            groq_model = st.selectbox(
                "Groq Model",
                options=["deepseek-r1-distill-llama-70b", "llama3-70b-8192", "mixtral-8x7b-32768","llama-3.3-70b-versatile"],
                help="Select which Groq model to use"
            )
            model_kwargs = {"model": groq_model}
        elif provider == "openai":
            openai_model = st.selectbox(
                "OpenAI Model",
                options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                help="Select which OpenAI model to use"
            )
            model_kwargs = {"model": openai_model}
        else:
            model_kwargs = {}

# Main content area
url = st.text_input(
    "Enter a web page URL:",
    placeholder="https://example.com",
    help="Enter the full URL including https://"
)

# Process the URL when the button is clicked
if st.button("Summarize", type="primary"):
    if not url:
        st.error("Please enter a valid URL")
    else:
        try:
            # Show a spinner while processing
            with st.spinner("Fetching and analyzing the web page..."):
                # Step 1: Initialize the LLM and embedding model
                llm = get_llm(provider, temperature=temperature, model_kwargs=model_kwargs)
                embed_model = get_embeddings(embedder)
                vectorstore = create_faiss_store(embed_model)
                
                # Step 2: Fetch the content from the URL
                raw_text = fetch_url_content(url)
                
                # Step 3: Generate a summary using the LLM
                summary = summarize_text(
                    text=raw_text,
                    llm=llm,
                    style=summary_style,
                    max_chars=max_chars
                )
                
                # Step 4: Store the summary in the vector database
                metadata = {
                    "source_url": url,
                    "timestamp": datetime.now().isoformat(),
                    "provider": provider,
                    "style": summary_style
                }
                store_summary(vectorstore, summary, metadata)
                
                # Step 5: Save to disk if requested
                if save_to_disk:
                    save_dir = os.path.join("data", "summaries")
                    os.makedirs(save_dir, exist_ok=True)
                    save_vectorstore(vectorstore, save_dir)
                    st.success(f"Summary saved to disk in {save_dir}")
            
            # Display the results
            st.subheader("📝 Summary")
            st.write(summary)
            
            # Show some stats
            st.info(f"Processed {len(raw_text)} characters from the webpage")
            
        except ValueError as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Error processing URL {url}: {e}")
        
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            logger.exception(f"Unexpected error processing URL {url}")

# Footer
st.markdown("---")
st.markdown(
    "Made with ❤️ using [Streamlit](https://streamlit.io) and [LangChain](https://langchain.com)"
)
