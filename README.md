# 🔍 Agentic Web Summarizer

A beginner-friendly application that demonstrates the concept of AI agents by summarizing web content using Large Language Models (LLMs).

## 📋 Overview

This project showcases how to build an agentic AI application that:

1. **Fetches content** from any web page
2. **Summarizes** the content using an LLM (OpenAI or Groq)
3. **Stores** the summaries in a vector database for potential future retrieval

The application is built with Streamlit for the user interface and LangChain for orchestrating the AI components.

## 🧠 What is an Agentic AI?

An "agent" in AI refers to a system that can:
- Perceive its environment (in this case, web pages)
- Make decisions based on that information (generate summaries)
- Take actions to achieve specific goals (store information for later use)

This simple application demonstrates these concepts in an approachable way.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- API keys for the LLM providers you want to use (OpenAI and/or Groq)

### Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd agentic_web_summarizer
   ```

2. Create a virtual environment:
   ```
   python -m venv ai-env
   ```

3. Activate the virtual environment:
   - Windows: `ai-env\Scripts\activate`
   - macOS/Linux: `source ai-env/bin/activate`

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Set up your API keys in one of these ways:

   a. As environment variables:
      - For OpenAI: `export OPENAI_API_KEY=your_api_key_here`
      - For Groq: `export GROQ_API_KEY=your_api_key_here`
      - For LangChain tracing: `export LANGCHAIN_API_KEY=your_api_key_here`
   
   b. Through the UI:
      - Launch the app and enter your API keys in the sidebar under "API Keys"
      - These keys are stored in session state and not saved to disk
   
   c. Create a .env file:
      ```
      OPENAI_API_KEY=your_openai_key
      GROQ_API_KEY=your_groq_key
      LANGCHAIN_API_KEY=your_langchain_key
      ```

### Running the Application

Start the Streamlit app:
```
streamlit run main.py
```

The application will be available at http://localhost:8501

## 🧩 Project Structure

```
agentic_web_summarizer/
├── agents/             # Agent definitions
│   └── web_agent.py    # Web agent implementation
├── llm/                # LLM provider integrations
│   └── llm_provider.py # LLM provider abstraction
├── tools/              # Tools used by the agents
│   ├── fetcher.py      # Web content fetching
│   ├── summarizer.py   # Text summarization
│   └── vectorstore.py  # Vector database operations
├── data/               # Directory for storing summaries (created at runtime)
├── main.py             # Main Streamlit application
└── requirements.txt    # Project dependencies
```

## 🔧 Key Components Explained

### LLM Providers

The application supports multiple LLM providers through the `llm_provider.py` module:
- **OpenAI**: Uses GPT models for high-quality summaries
- **Groq**: An alternative provider that offers fast inference

The application also supports LangChain tracing for monitoring and debugging:
- Track token usage and costs
- Debug prompts and responses
- Analyze chain execution
- Monitor performance metrics

To enable tracing, provide a LangChain API key in the UI or as an environment variable.

### Tools

The application uses several tools:

1. **Fetcher**: Retrieves and cleans content from web pages
2. **Summarizer**: Generates concise summaries using LLMs
3. **VectorStore**: Stores and retrieves summaries using vector embeddings

### Agents

The `web_agent.py` module demonstrates how to build more complex agents that can use tools to accomplish tasks.

## 🔄 How It Works

1. The user enters a URL in the Streamlit interface
2. The application fetches the content from the URL
3. The content is sent to an LLM with a prompt to generate a summary
4. The summary is displayed to the user and stored in a vector database
5. (Optional) The summary can be saved to disk for future reference

## 🛠️ Customization

You can customize the application by:
- Adding new LLM providers in `llm_provider.py`
- Creating new summary styles in `summarizer.py`
- Extending the agent capabilities in `web_agent.py`

## 📚 Learning Resources

To learn more about the concepts used in this project:

- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.