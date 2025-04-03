"""
LLM Service module for handling connections to LLM providers.

This module provides functions for configuring and managing connections to LLM providers
such as OpenAI and Anthropic. It handles the authentication, request formatting, and 
response parsing for these services.
"""

import os
import logging
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from enum import Enum

# Import OpenAI and Anthropic clients
from openai import OpenAI
from anthropic import Anthropic

# Set up logging
logger = logging.getLogger(__name__)

# Define constants
DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_CLAUDE_MODEL = "claude-3-opus-20240229"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 4000

class LLMProvider(Enum):
    """Enum for supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


def get_openai_client() -> OpenAI:
    """
    Get an authenticated OpenAI client using environment variables.
    
    Returns:
        OpenAI: Authenticated OpenAI client
    
    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    return OpenAI(api_key=api_key)


def get_anthropic_client() -> Anthropic:
    """
    Get an authenticated Anthropic client using environment variables.
    
    Returns:
        Anthropic: Authenticated Anthropic client
    
    Raises:
        ValueError: If the ANTHROPIC_API_KEY environment variable is not set
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
    
    return Anthropic(api_key=api_key)


def get_llm_client(provider: LLMProvider = LLMProvider.OPENAI) -> Union[OpenAI, Anthropic]:
    """
    Get an authenticated LLM client based on the specified provider.
    
    Args:
        provider: The LLM provider to use (default: OpenAI)
    
    Returns:
        Union[OpenAI, Anthropic]: Authenticated LLM client
    
    Raises:
        ValueError: If the API key environment variable is not set
    """
    if provider == LLMProvider.OPENAI:
        return get_openai_client()
    elif provider == LLMProvider.ANTHROPIC:
        return get_anthropic_client()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_openai_response(
    messages: List[Dict[str, str]],
    model: str = "gpt-4",
    temperature: float = 0.7,
    max_tokens: int = 1500,
    functions: Optional[List[Dict[str, Any]]] = None,
    retry_count: int = 3,
    retry_delay: float = 2.0
) -> Any:
    """
    Generate a response from the OpenAI API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model to use (defaults to gpt-4)
        temperature: Temperature for response generation
        max_tokens: Maximum tokens in the response
        functions: List of function definitions for function calling
        retry_count: Number of times to retry on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        OpenAI API response object
    """
    # Check if API key is set
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        logger.error("OpenAI API key is not set in environment variables")
        raise ValueError("OpenAI API key is not set. Please set the OPENAI_API_KEY environment variable.")
    
    logger.info(f"Making OpenAI API call with model {model} and {len(messages)} messages")
    
    client = OpenAI(api_key=api_key)
    
    # Format messages for OpenAI
    formatted_messages = []
    for msg in messages:
        formatted_msg = {
            "role": msg["role"],
            "content": str(msg["content"]) if msg["content"] is not None else ""
        }
        
        # Add 'name' for function messages if present
        if msg["role"] == "function" and "name" in msg:
            formatted_msg["name"] = msg["name"]
        
        formatted_messages.append(formatted_msg)
    
    # Setup request parameters
    kwargs = {
        "model": model,
        "messages": formatted_messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    # Add functions if provided
    if functions:
        logger.info(f"Including {len(functions)} function definitions")
        kwargs["tool_choice"] = "auto"
        kwargs["tools"] = [{"type": "function", "function": fn} for fn in functions]
    
    # Try API call with retries
    for attempt in range(retry_count):
        try:
            logger.debug(f"Attempt {attempt + 1}/{retry_count} to call OpenAI API")
            response = client.chat.completions.create(**kwargs)
            logger.info(f"OpenAI API call successful")
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API (attempt {attempt + 1}/{retry_count}): {str(e)}")
            if attempt < retry_count - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All retry attempts failed")
                raise


def generate_anthropic_response(
    messages: List[Dict[str, str]],
    model: str = "claude-3-haiku-20240307",
    temperature: float = 0.7,
    max_tokens: int = 1500,
    retry_count: int = 3,
    retry_delay: float = 2.0
) -> Any:
    """
    Generate a response from the Anthropic API.
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model to use (defaults to claude-3-haiku-20240307)
        temperature: Temperature for response generation
        max_tokens: Maximum tokens in the response
        retry_count: Number of times to retry on failure
        retry_delay: Delay between retries in seconds
        
    Returns:
        Anthropic API response object
    """
    # Check if API key is set
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        logger.error("Anthropic API key is not set in environment variables")
        raise ValueError("Anthropic API key is not set. Please set the ANTHROPIC_API_KEY environment variable.")
    
    logger.info(f"Making Anthropic API call with model {model} and {len(messages)} messages")
    
    client = Anthropic(api_key=api_key)
    
    # Format messages for Anthropic 
    formatted_messages = []
    for msg in messages:
        role = "assistant" if msg["role"] == "assistant" else "user"
        content = str(msg["content"]) if msg["content"] is not None else ""
        
        formatted_messages.append({
            "role": role,
            "content": content
        })
    
    # Try API call with retries
    for attempt in range(retry_count):
        try:
            logger.debug(f"Attempt {attempt + 1}/{retry_count} to call Anthropic API")
            response = client.messages.create(
                model=model,
                messages=formatted_messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info(f"Anthropic API call successful")
            return response
        except Exception as e:
            logger.error(f"Error calling Anthropic API (attempt {attempt + 1}/{retry_count}): {str(e)}")
            if attempt < retry_count - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                logger.error("All retry attempts failed")
                raise


def generate_response(
    messages: List[Dict[str, str]],
    functions: Optional[List[Dict[str, Any]]] = None,
    provider: LLMProvider = LLMProvider.OPENAI,
    model: Optional[str] = None,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS
) -> str:
    """
    Generate a response from the specified LLM provider.
    
    Args:
        messages: List of message objects (role, content)
        functions: Optional list of function definitions
        provider: LLM provider to use (default: OpenAI)
        model: Model to use (default: provider-specific default)
        temperature: Temperature parameter (default: 0.1)
        max_tokens: Maximum tokens in response (default: 1000)
    
    Returns:
        str: Text response from the LLM
    """
    try:
        if provider == LLMProvider.OPENAI:
            # Set default model if not specified
            if not model:
                model = DEFAULT_MODEL
            
            # Get response from OpenAI
            response = generate_openai_response(
                messages=messages,
                functions=functions,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract text content from response
            if hasattr(response, "choices") and response.choices:
                return response.choices[0].message.content or ""
            return ""
        
        elif provider == LLMProvider.ANTHROPIC:
            # Set default model if not specified
            if not model:
                model = DEFAULT_CLAUDE_MODEL
            
            # Get response from Anthropic
            response = generate_anthropic_response(
                messages=messages,
                functions=functions,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            # Extract text content from response
            if hasattr(response, "content") and response.content:
                for content_block in response.content:
                    if content_block.type == "text":
                        return content_block.text
            return ""
        
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    except Exception as e:
        logger.error(f"Error generating response from {provider}: {e}")
        return f"Error generating response: {str(e)}" 