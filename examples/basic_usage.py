# examples/basic_usage.py
"""
Basic usage examples for the Agentic Web Summarizer.

This script demonstrates how to use the core components of the
Agentic Web Summarizer in your own Python code.
"""

import os
import sys
from pathlib import Path

# Add the parent directory to the path so we can import our modules
parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

# Import our modules
from llm.llm_provider import get_llm
from tools.fetcher import fetch_url_content
from tools.summarizer import summarize_text
from tools.vectorstore import get_embeddings, create_faiss_store, store_summary
from agents.summarization_agent import SummarizationAgent

def example_1_basic_usage():
    """
    Example 1: Basic usage - fetch and summarize a web page.
    """
    print("\n=== Example 1: Basic Usage ===\n")
    
    # Step 1: Initialize the LLM
    llm = get_llm("openai")
    
    # Step 2: Fetch content from a URL
    url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
    print(f"Fetching content from: {url}")
    content = fetch_url_content(url)
    print(f"Fetched {len(content)} characters")
    
    # Step 3: Summarize the content
    print("Generating summary...")
    summary = summarize_text(content, llm)
    
    # Step 4: Print the summary
    print("\nSummary:")
    print("-" * 40)
    print(summary)
    print("-" * 40)

def example_2_vector_storage():
    """
    Example 2: Store summaries in a vector database.
    """
    print("\n=== Example 2: Vector Storage ===\n")
    
    # Step 1: Initialize the LLM and embedding model
    llm = get_llm("openai")
    embed_model = get_embeddings("openai")
    
    # Step 2: Create a vector store
    vectorstore = create_faiss_store(embed_model)
    
    # Step 3: Fetch and summarize content
    urls = [
        "https://en.wikipedia.org/wiki/Machine_learning",
        "https://en.wikipedia.org/wiki/Deep_learning"
    ]
    
    for url in urls:
        print(f"Processing: {url}")
        content = fetch_url_content(url)
        summary = summarize_text(content, llm)
        
        # Step 4: Store the summary with metadata
        metadata = {"source_url": url, "topic": "AI"}
        store_summary(vectorstore, summary, metadata)
        print(f"Stored summary for: {url}")
    
    print("\nSummaries stored in vector database")

def example_3_agent_based():
    """
    Example 3: Using the agent-based approach.
    """
    print("\n=== Example 3: Agent-Based Approach ===\n")
    
    # Step 1: Initialize the LLM
    llm = get_llm("openai")
    
    # Step 2: Create a summarization agent
    agent = SummarizationAgent(llm, verbose=True)
    
    # Step 3: Run the agent on a URL
    url = "https://en.wikipedia.org/wiki/Natural_language_processing"
    print(f"Running agent on: {url}")
    
    result = agent.run(url, {"topic": "NLP"})
    
    # Step 4: Print the summary
    print("\nAgent-generated summary:")
    print("-" * 40)
    print(result["summary"])
    print("-" * 40)

def main():
    """Run all examples."""
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    try:
        # Run the examples
        example_1_basic_usage()
        example_2_vector_storage()
        example_3_agent_based()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()