"""
LLM Integration Package for the Adoption Rate Chatbot.

This package provides functionality for integrating Large Language Models (LLMs)
into the Adoption Rate Chatbot, including API interactions, prompt templates,
conversation context management, function calling, and response formatting.
"""

# Import key functions from modules
from .llm_service import (
    get_llm_client,
    get_openai_client,
    get_anthropic_client,
    generate_response,
    LLMProvider
)

from .prompts import (
    get_system_prompt,
    get_descriptive_prompt_template,
    get_diagnostic_prompt_template,
    get_predictive_prompt_template,
    get_prescriptive_prompt_template,
    determine_prompt_type
)

from .context_manager import (
    MessageHistory,
    add_message,
    trim_context_to_max_tokens,
    get_context_window,
    add_data_to_context,
    add_chart_reference
)

from .function_calling import (
    get_function_definitions,
    format_function_call,
    execute_function,
    handle_function_response
)

from .response_formatter import (
    format_response,
    add_chart_references,
    add_chart_reference_to_response,
    format_numbers,
    format_key_metrics,
    format_anomalies,
    format_date_references
)

# Define public API
__all__ = [
    # LLM Service
    'get_llm_client',
    'get_openai_client',
    'get_anthropic_client',
    'generate_response',
    'LLMProvider',
    
    # Prompts
    'get_system_prompt',
    'get_descriptive_prompt_template',
    'get_diagnostic_prompt_template',
    'get_predictive_prompt_template',
    'get_prescriptive_prompt_template',
    'determine_prompt_type',
    
    # Context Manager
    'MessageHistory',
    'add_message',
    'trim_context_to_max_tokens',
    'get_context_window',
    'add_data_to_context',
    'add_chart_reference',
    
    # Function Calling
    'get_function_definitions',
    'format_function_call',
    'execute_function',
    'handle_function_response',
    
    # Response Formatter
    'format_response',
    'add_chart_references',
    'add_chart_reference_to_response',
    'format_numbers',
    'format_key_metrics',
    'format_anomalies',
    'format_date_references'
] 