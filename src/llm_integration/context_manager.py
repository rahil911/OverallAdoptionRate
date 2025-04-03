"""
Context Manager module for handling conversation history and context.

This module provides functionality for managing conversation history,
tracking context, and ensuring the context stays within token limits.
"""

from typing import Dict, List, Optional, Union, Any
import json
import tiktoken
import logging

# Set up logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = "gpt-4o-mini"
MAX_TOKENS = 256000
MAX_CONTEXT_TOKENS = 200000
RESERVED_TOKENS = 8000


class MessageHistory:
    """Class for managing conversation message history"""
    
    def __init__(self, system_prompt: str = "", model: str = DEFAULT_MODEL):
        """
        Initialize message history with optional system prompt.
        
        Args:
            system_prompt: Optional system prompt to provide context
            model: Model name for token counting purposes
        """
        self.messages = []
        self.model = model
        self.tokenizer = tiktoken.encoding_for_model(model) if model.startswith("gpt") else None
        
        # Add system message if provided
        if system_prompt:
            self.add_message("system", system_prompt)
    
    def add_message(self, role: str, content: Any, name: Optional[str] = None) -> None:
        """
        Add a message to the history.
        
        Args:
            role: Message role (e.g., "system", "user", "assistant", "function")
            content: Message content (will be converted to string if not already)
            name: Optional name field, required for function messages
        """
        # Ensure content is a string if it's not a dictionary
        if content is None:
            content = ""
        elif isinstance(content, dict) and 'content' in content:
            # Handle cases where content is passed as a message dict
            content = content.get('content', '')
            if content is None:
                content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        # Create message dict
        message = {"role": role, "content": content}
        
        # Add name if provided (required for function messages)
        if name is not None:
            message["name"] = name
        
        # Add to messages
        self.messages.append(message)
        
        logger.debug(f"Added message: {role} - {str(content)[:30]}...")
        
        # Check if we need to trim the context
        tokens = self.count_tokens()
        if tokens > MAX_CONTEXT_TOKENS:
            logger.info(f"Context exceeds token limit ({tokens} > {MAX_CONTEXT_TOKENS}). Trimming...")
            self.trim_context_to_max_tokens()
    
    def get_messages(self) -> List[Dict[str, str]]:
        """
        Get all messages in the history.
        
        Returns:
            List[Dict[str, str]]: List of message dictionaries
        """
        return self.messages
    
    def count_tokens(self) -> int:
        """
        Count the number of tokens in the message history.
        
        Returns:
            int: Number of tokens
        """
        token_count = 0
        
        for message in self.messages:
            # Add 4 tokens for message format
            token_count += 4
            
            # Add tokens for each message content
            for key, value in message.items():
                # Ensure value is a string before encoding
                if value is None:
                    str_value = ""
                elif not isinstance(value, str):
                    str_value = str(value)
                else:
                    str_value = value
                    
                token_count += len(self.tokenizer.encode(str_value))
                
                # Add 1 token for each key-value pair
                token_count += 1
        
        # Add 2 tokens for formatting
        token_count += 2
        
        return token_count
    
    def trim_context_to_max_tokens(self, max_tokens: int = MAX_CONTEXT_TOKENS) -> None:
        """
        Trim the conversation history to stay within token limits.
        Keeps system prompt and most recent messages, removing older messages.
        
        Args:
            max_tokens: Maximum number of tokens to keep
        """
        if len(self.messages) <= 2:
            # If we only have system and one user message, nothing to trim
            return
        
        # Save system message(s) if any
        system_messages = [m for m in self.messages if m["role"] == "system"]
        non_system_messages = [m for m in self.messages if m["role"] != "system"]
        
        # Keep removing oldest non-system messages until we're under the limit
        while non_system_messages and (self.count_tokens() > max_tokens):
            # Remove the oldest non-system message
            removed_message = non_system_messages.pop(0)
            self.messages.remove(removed_message)
            logger.info(f"Removed message: {removed_message['role']} - {removed_message['content'][:30]}...")
    
    def clear_history(self, keep_system: bool = True) -> None:
        """
        Clear the message history.
        
        Args:
            keep_system: Whether to keep system messages (default: True)
        """
        if keep_system:
            system_messages = [m for m in self.messages if m["role"] == "system"]
            self.messages = system_messages
        else:
            self.messages = []


def add_message(message_history: MessageHistory, role: str, content: Any, name: Optional[str] = None) -> MessageHistory:
    """
    Add a message to the message history.
    
    Args:
        message_history: MessageHistory object
        role: Message role (e.g., "system", "user", "assistant", "function")
        content: Message content (will be converted to string if not already)
        name: Optional name field, required for function messages
    
    Returns:
        MessageHistory: Updated message history
    """
    message_history.add_message(role, content, name)
    return message_history


def trim_context_to_max_tokens(message_history: MessageHistory, max_tokens: int = MAX_CONTEXT_TOKENS) -> MessageHistory:
    """
    Trim the conversation history to stay within token limits.
    
    Args:
        message_history: MessageHistory object
        max_tokens: Maximum number of tokens to keep
    
    Returns:
        MessageHistory: Trimmed message history
    """
    message_history.trim_context_to_max_tokens(max_tokens)
    return message_history


def get_context_window(message_history: MessageHistory) -> List[Dict[str, str]]:
    """
    Get the current context window (all messages).
    
    Args:
        message_history: MessageHistory object
    
    Returns:
        List[Dict[str, str]]: List of messages in the context window
    """
    return message_history.get_messages()


def add_data_to_context(message_history: MessageHistory, data: Dict[str, Any], role: str = "system") -> MessageHistory:
    """
    Add data (e.g., chart data, analysis results) to the context.
    
    Args:
        message_history: MessageHistory object
        data: Data to add to the context
        role: Role for the added message (default: "system")
    
    Returns:
        MessageHistory: Updated message history
    """
    data_str = json.dumps(data, indent=2)
    context_message = f"Reference data for your response:\n{data_str}"
    message_history.add_message(role, context_message)
    return message_history


def add_chart_reference(message_history: MessageHistory, chart_description: str, chart_path: str) -> MessageHistory:
    """
    Add a reference to a chart or visualization to the context.
    
    Args:
        message_history: MessageHistory object
        chart_description: Description of what the chart shows
        chart_path: Path to the chart file
    
    Returns:
        MessageHistory: Updated message history
    """
    chart_reference = f"The following chart may be helpful for your response:\n\nChart: {chart_description}\nPath: {chart_path}"
    message_history.add_message("system", chart_reference)
    return message_history 