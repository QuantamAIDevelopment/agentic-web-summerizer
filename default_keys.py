# default_keys.py
"""
This module contains default API keys for demonstration purposes.
These keys should be used only when users don't provide their own keys.

WARNING: In a production environment, you should never hardcode API keys.
         These are provided only for demonstration and testing purposes.

How to use:
1. Replace the placeholder keys below with your actual API keys
2. Set USE_DEFAULT_KEYS to True to enable the use of default keys
3. Users can still override these keys by providing their own in the UI

Security considerations:
- These keys will be visible to anyone with access to your source code
- Consider using environment variables or a secure key management system in production
- For open source projects, never commit real API keys to version control
"""

# Default API keys (replace with your actual keys for testing)
DEFAULT_OPENAI_API_KEY = "sk-demo-openai-key-for-testing-purposes-only"
DEFAULT_GROQ_API_KEY = "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"
DEFAULT_LANGCHAIN_API_KEY = "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"

# Flag to enable/disable the use of default keys
# Set to True to enable using default keys when user keys are not provided
# Set to False to require users to provide their own keys
USE_DEFAULT_KEYS = True