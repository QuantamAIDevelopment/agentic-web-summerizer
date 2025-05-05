# tools/summarizer.py

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.llm import LLMChain
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_PROMPT_TEMPLATE = """
You are an expert at summarizing long-form web content.

Summarize the following content in a clear, concise way. Focus on the key points and skip irrelevant details.

{text}

Summary:
"""

def summarize_text(text: str, llm, style: str = "default", max_chars: int = 4000) -> str:
    
    if len(text) > max_chars:
        logger.warning(f"Input text exceeds {max_chars} characters. Truncating.")
        text = text[:max_chars]

    prompt_template = DEFAULT_PROMPT_TEMPLATE

    if style == "bullet":
        prompt_template = """
Summarize the following content as a list of bullet points, focusing on the most important facts and themes:

{text}

Bullet Point Summary:
"""

    try:
        prompt = PromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        summary = chain.run(text=text)
        logger.info("Summary generation complete.")
        return summary.strip()
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise RuntimeError(f"Failed to summarize text: {e}")
