# 🎓 Agentic Web Summarizer Tutorial

This tutorial will guide you through the Agentic Web Summarizer project, explaining the key concepts and how the code works. It's designed for beginners who want to understand how to build AI agents.

## 📚 What You'll Learn

- What an AI agent is and how it works
- How to use LangChain to build AI applications
- How to integrate with LLMs like OpenAI and Groq
- How to store and retrieve vector embeddings
- How to build a simple web interface with Streamlit

## 🧩 Project Components

### 1. The Main Application (`main.py`)

The `main.py` file is the entry point for the Streamlit web application. It:

1. Creates a user interface with Streamlit
2. Takes a URL input from the user
3. Orchestrates the process of fetching, summarizing, and storing content

```python
# This is the main flow of the application
if st.button("Summarize"):
    # Step 1: Initialize the LLM and embedding model
    llm = get_llm(provider, temperature=temperature)
    embed_model = get_embeddings(embedder)
    vectorstore = create_faiss_store(embed_model)
    
    # Step 2: Fetch the content from the URL
    raw_text = fetch_url_content(url)
    
    # Step 3: Generate a summary using the LLM
    summary = summarize_text(text=raw_text, llm=llm, style=summary_style)
    
    # Step 4: Store the summary in the vector database
    store_summary(vectorstore, summary, {"source_url": url})
```

### 2. LLM Provider (`llm/llm_provider.py`)

This module abstracts away the details of different LLM providers:

```python
def get_llm(provider: str = 'openai', temperature: float = 0.0):
    """Returns a language model instance based on the specified provider."""
    if provider == 'groq':
        return GroqChatModel(temperature=temperature, model="deepseek-r1-distill-llama-70b")
    elif provider == 'openai':
        return ChatOpenAI(temperature=temperature)
```

### 3. Web Content Fetcher (`tools/fetcher.py`)

This tool fetches and cleans content from web pages:

```python
def fetch_url_content(url: str, timeout: int = 10) -> str:
    """Fetches and extracts the main content from a web page."""
    # Fetch the page
    response = requests.get(url, headers=headers, timeout=timeout)
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(response.text, "html.parser")
    
    # Clean and extract text
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    
    return clean_text
```

### 4. Summarizer (`tools/summarizer.py`)

This tool uses an LLM to generate summaries:

```python
def summarize_text(text: str, llm, style: str = "default", max_chars: int = 4000) -> str:
    """Summarizes text using an LLM."""
    # Truncate if needed
    if len(text) > max_chars:
        text = text[:max_chars]
    
    # Create a prompt
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Run the LLM chain
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(text=text)
    
    return summary.strip()
```

### 5. Vector Store (`tools/vectorstore.py`)

This module handles storing and retrieving summaries using vector embeddings:

```python
def store_summary(vectorstore: FAISS, summary: str, metadata: dict):
    """Stores a summary in the vector store with associated metadata."""
    vectorstore.add_texts([summary], [metadata])
```

### 6. Agents (`agents/summarization_agent.py`)

This module shows how to build a more complex agent that can reason about tasks:

```python
class SummarizationAgent:
    """An agent that can fetch web content, summarize it, and store the results."""
    
    def __init__(self, llm, embedding_provider="openai", verbose=False):
        # Initialize tools and agent
        self.llm = llm
        self.agent = self._create_agent()
    
    def run(self, url: str, metadata=None):
        """Run the agent to fetch and summarize content from a URL."""
        # Fetch content
        raw_text = fetch_url_content(url)
        
        # Summarize
        summary = summarize_text(raw_text, self.llm)
        
        # Store
        store_summary(self.vectorstore, summary, metadata)
        
        return {"summary": summary, ...}
```

## 🔄 The Agent Workflow

Here's how the components work together:

1. The user enters a URL in the Streamlit interface
2. The application fetches the content using `fetch_url_content()`
3. The content is summarized using `summarize_text()` and an LLM
4. The summary is stored in a vector database using `store_summary()`
5. The summary is displayed to the user

## 🧠 Understanding AI Agents

An AI agent is a system that can:

1. **Perceive** its environment (e.g., read web pages)
2. **Reason** about what to do (e.g., decide how to summarize)
3. **Act** to achieve goals (e.g., generate and store summaries)

In this project:
- The **perception** happens through the web fetcher
- The **reasoning** happens in the LLM
- The **action** happens when we generate and store summaries

## 🔍 Vector Embeddings Explained

Vector embeddings are numerical representations of text that capture semantic meaning. In this project:

1. We convert summaries to vector embeddings using models like OpenAI's embeddings
2. We store these vectors in a FAISS database for efficient retrieval
3. This allows us to find similar summaries based on meaning, not just keywords

## 🛠️ Extending the Project

Here are some ways you could extend this project:

1. Add a search feature to find similar summaries
2. Support more LLM providers
3. Add the ability to summarize multiple pages at once
4. Create a more complex agent that can follow links and summarize entire websites
5. Add a feature to compare summaries from different LLMs

## 📝 Exercises for Learning

1. Modify the summarizer to support a new summary style (e.g., "academic" or "ELI5")
2. Add a new LLM provider to the `llm_provider.py` module
3. Create a new tool that extracts images from web pages
4. Modify the agent to follow links on a page and summarize those as well
5. Add a feature to the Streamlit app to view previously stored summaries

## 🔗 Further Resources

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [OpenAI API Documentation](https://platform.openai.com/docs/introduction)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)