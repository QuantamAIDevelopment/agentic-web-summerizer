# agents/web_agent.py

from typing import List, Optional
import logging

from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.chat_models.base import BaseChatModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def build_agent(
    llm: BaseChatModel,
    tools: List[Tool],
    agent_type: str = "zero-shot-react-description",
    agent_name: Optional[str] = None,
    verbose: bool = True
):
    """
    Initializes a LangChain agent with given tools and language model.

    Args:
        llm (BaseChatModel): A LangChain-compatible chat model.
        tools (List[Tool]): List of LangChain tools.
        agent_type (str): Type of agent to use. Defaults to "zero-shot-react-description".
        agent_name (str): Optional name for logging or debugging.
        verbose (bool): Whether to show reasoning steps.

    Returns:
        AgentExecutor: An initialized LangChain agent.

    Raises:
        ValueError: If invalid agent type is provided.
    """
    supported_agent_types = {
        "zero-shot-react-description": AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        "openai-functions": AgentType.OPENAI_FUNCTIONS,
        "chat-conversational-react-description": AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    }

    if agent_type not in supported_agent_types:
        raise ValueError(f"Unsupported agent type: {agent_type}")

    if agent_name:
        logger.info(f"Building agent: {agent_name} with type: {agent_type}")

    return initialize_agent(
        tools=tools,
        llm=llm,
        agent_type=supported_agent_types[agent_type],
        verbose=verbose
    )
