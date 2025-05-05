# agents/summarization_agent.py
"""
This module implements a summarization agent that can fetch and summarize web content.

The agent uses LangChain's agent framework to orchestrate multiple tools:
1. A web fetching tool to retrieve content
2. A summarization tool to generate summaries
3. A storage tool to save summaries

This demonstrates how to build more complex agentic workflows.
"""

import logging
from typing import Optional, List, Dict, Any

from langchain.agents import Tool, AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain.agents import initialize_agent
from langchain.chat_models.base import BaseChatModel

from tools.fetcher import fetch_url_content
from tools.summarizer import summarize_text
from tools.vectorstore import store_summary, get_embeddings, create_faiss_store

logger = logging.getLogger(__name__)

class SummarizationAgent:
    """
    An agent that can fetch web content, summarize it, and store the results.
    
    This class demonstrates how to build a simple agent that combines multiple
    tools to accomplish a task.
    """
    
    def __init__(
        self, 
        llm: BaseChatModel,
        embedding_provider: str = "openai",
        verbose: bool = False
    ):
        """
        Initialize the summarization agent.
        
        Args:
            llm: The language model to use for summarization and agent reasoning
            embedding_provider: The embedding provider to use for vector storage
            verbose: Whether to show the agent's reasoning steps
        """
        self.llm = llm
        self.verbose = verbose
        self.embedding_provider = embedding_provider
        
        # Initialize embedding model and vector store
        self.embed_model = get_embeddings(embedding_provider)
        self.vectorstore = create_faiss_store(self.embed_model)
        
        # Create the agent with tools
        self.agent = self._create_agent()
        
    def _create_agent(self) -> AgentExecutor:
        """
        Create a LangChain agent with the necessary tools.
        
        Returns:
            An initialized agent executor
        """
        # Define the tools the agent can use
        tools = [
            Tool(
                name="FetchWebContent",
                func=fetch_url_content,
                description="Fetches and extracts the main content from a web page. Input should be a valid URL."
            ),
            Tool(
                name="SummarizeText",
                func=lambda text: summarize_text(text, self.llm),
                description="Summarizes a piece of text. Input should be the text to summarize."
            )
        ]
        
        # Initialize the agent
        return initialize_agent(
            tools=tools,
            llm=self.llm,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose
        )
    
    def run(self, url: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run the agent to fetch and summarize content from a URL.
        
        Args:
            url: The URL to fetch and summarize
            metadata: Optional metadata to store with the summary
            
        Returns:
            A dictionary containing the raw text, summary, and metadata
        """
        logger.info(f"Running summarization agent on URL: {url}")
        
        # Step 1: Fetch the content
        raw_text = fetch_url_content(url)
        
        # Step 2: Summarize the content
        summary = summarize_text(raw_text, self.llm)
        
        # Step 3: Store the summary with metadata
        if metadata is None:
            metadata = {}
        
        metadata["source_url"] = url
        store_summary(self.vectorstore, summary, metadata)
        
        return {
            "raw_text": raw_text,
            "summary": summary,
            "metadata": metadata
        }
    
    def run_with_agent(self, query: str) -> str:
        """
        Run the agent with a natural language query.
        
        This method demonstrates how to use the LangChain agent framework
        to handle more complex, reasoning-based tasks.
        
        Args:
            query: A natural language query like "Summarize the content from https://example.com"
            
        Returns:
            The agent's response
        """
        logger.info(f"Running agent with query: {query}")
        return self.agent.run(query)


# Example usage
if __name__ == "__main__":
    from llm.llm_provider import get_llm
    
    # Initialize the agent
    llm = get_llm("openai")
    agent = SummarizationAgent(llm, verbose=True)
    
    # Run the agent on a URL
    result = agent.run("https://en.wikipedia.org/wiki/Artificial_intelligence")
    print(f"Summary: {result['summary']}")
    
    # Or run with a natural language query
    response = agent.run_with_agent("Summarize the content from https://en.wikipedia.org/wiki/Machine_learning")
    print(f"Agent response: {response}")