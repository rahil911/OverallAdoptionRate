# LLM Integration Documentation

## Overview
The LLM (Large Language Model) integration layer provides the core natural language processing capabilities for the chatbot. This layer manages interactions with LLM services (OpenAI and Anthropic), handles prompt engineering, manages conversation context, and implements function calling for data access and analysis.

## Directory Structure
```
src/llm_integration/
├── __init__.py
├── llm_service.py
├── prompts.py
├── context_manager.py
├── function_calling.py
├── query_processor.py
└── response_formatter.py
```

## Components

### 1. LLM Service (`llm_service.py`)

#### Purpose
Manages interactions with LLM providers (OpenAI and Anthropic) and handles API communication.

#### Key Features
- Multi-provider support (OpenAI, Anthropic)
- API key management
- Request/response handling
- Error recovery and retry logic
- Rate limiting compliance

#### Core Classes and Methods

##### LLMService
```python
class LLMService:
    def generate_response(
        messages: List[Dict],
        functions: List[Dict] = None,
        provider: str = "openai"
    ) -> Dict:
        """
        Generates response from LLM
        
        Parameters:
            messages: List of conversation messages
            functions: List of available functions
            provider: LLM provider to use
            
        Returns:
            Dictionary containing LLM response
        """

    def get_openai_client() -> OpenAI:
        """
        Gets authenticated OpenAI client
        
        Returns:
            OpenAI client instance
        """

    def get_anthropic_client() -> Anthropic:
        """
        Gets authenticated Anthropic client
        
        Returns:
            Anthropic client instance
        """

    def handle_api_error(
        error: Exception,
        context: Dict
    ) -> Dict:
        """
        Handles API errors with retry logic
        
        Parameters:
            error: Exception object
            context: Error context
            
        Returns:
            Error handling result
        """
```

### 2. Prompt Engineering (`prompts.py`)

#### Purpose
Manages prompt templates and implements prompt engineering strategies for different query types.

#### Key Features
- Template management
- Dynamic prompt generation
- System prompt definition
- Context injection
- Template versioning

#### Core Classes and Methods

##### PromptManager
```python
class PromptManager:
    def get_system_prompt() -> str:
        """
        Gets the standard system prompt
        
        Returns:
            System prompt string
        """

    def get_prompt_template(
        query_type: str
    ) -> str:
        """
        Gets prompt template for query type
        
        Parameters:
            query_type: Type of query
            
        Returns:
            Prompt template string
        """

    def format_prompt(
        template: str,
        context: Dict
    ) -> str:
        """
        Formats prompt template with context
        
        Parameters:
            template: Prompt template
            context: Context variables
            
        Returns:
            Formatted prompt
        """

    def determine_prompt_type(
        query: str
    ) -> str:
        """
        Determines appropriate prompt type
        
        Parameters:
            query: User query
            
        Returns:
            Prompt type string
        """
```

### 3. Context Management (`context_manager.py`)

#### Purpose
Manages conversation history and maintains context across multiple interactions.

#### Key Features
- Message history management
- Token counting
- Context window management
- Context injection
- State persistence

#### Core Classes and Methods

##### ContextManager
```python
class ContextManager:
    def add_message(
        role: str,
        content: str,
        name: str = None
    ) -> None:
        """
        Adds message to conversation history
        
        Parameters:
            role: Message role
            content: Message content
            name: Optional name for function messages
        """

    def get_context_window(
        max_tokens: int = 4000
    ) -> List[Dict]:
        """
        Gets current context window
        
        Parameters:
            max_tokens: Maximum tokens to include
            
        Returns:
            List of context messages
        """

    def count_tokens(
        text: str
    ) -> int:
        """
        Counts tokens in text
        
        Parameters:
            text: Input text
            
        Returns:
            Token count
        """

    def trim_context(
        max_tokens: int
    ) -> None:
        """
        Trims context to fit token limit
        
        Parameters:
            max_tokens: Maximum tokens to keep
        """
```

### 4. Function Calling (`function_calling.py`)

#### Purpose
Implements function calling capabilities to allow LLM to access data and analysis functions.

#### Key Features
- Function definition management
- Parameter validation
- Function execution
- Result formatting
- Error handling

#### Core Classes and Methods

##### FunctionRegistry
```python
class FunctionRegistry:
    def get_function_definitions() -> List[Dict]:
        """
        Gets all available function definitions
        
        Returns:
            List of function definitions
        """

    def register_function(
        name: str,
        description: str,
        parameters: Dict
    ) -> None:
        """
        Registers new function
        
        Parameters:
            name: Function name
            description: Function description
            parameters: Parameter schema
        """

    def execute_function(
        name: str,
        parameters: Dict
    ) -> Any:
        """
        Executes registered function
        
        Parameters:
            name: Function name
            parameters: Function parameters
            
        Returns:
            Function result
        """
```

### 5. Query Processing (`query_processor.py`)

#### Purpose
Processes natural language queries and manages the complete query-response pipeline.

#### Key Features
- Intent recognition
- Entity extraction
- Query transformation
- Response generation
- Context awareness

#### Core Classes and Methods

##### QueryProcessor
```python
class QueryProcessor:
    def process_query(
        query: str,
        context: Dict = None
    ) -> Dict:
        """
        Processes user query
        
        Parameters:
            query: User query
            context: Optional context
            
        Returns:
            Processing result
        """

    def extract_entities(
        query: str
    ) -> Dict[str, List]:
        """
        Extracts entities from query
        
        Parameters:
            query: User query
            
        Returns:
            Dictionary of extracted entities
        """

    def classify_intent(
        query: str
    ) -> str:
        """
        Classifies query intent
        
        Parameters:
            query: User query
            
        Returns:
            Intent classification
        """

    def transform_query(
        query: str,
        entities: Dict,
        intent: str
    ) -> Dict:
        """
        Transforms query for processing
        
        Parameters:
            query: User query
            entities: Extracted entities
            intent: Query intent
            
        Returns:
            Transformed query
        """
```

### 6. Response Formatting (`response_formatter.py`)

#### Purpose
Formats LLM responses for consistency and readability.

#### Key Features
- Response standardization
- Number formatting
- Date formatting
- Chart references
- Markdown formatting

#### Core Classes and Methods

##### ResponseFormatter
```python
class ResponseFormatter:
    def format_response(
        response: Dict
    ) -> str:
        """
        Formats LLM response
        
        Parameters:
            response: Raw LLM response
            
        Returns:
            Formatted response string
        """

    def format_numbers(
        text: str
    ) -> str:
        """
        Formats numbers in text
        
        Parameters:
            text: Input text
            
        Returns:
            Text with formatted numbers
        """

    def format_dates(
        text: str
    ) -> str:
        """
        Formats dates in text
        
        Parameters:
            text: Input text
            
        Returns:
            Text with formatted dates
        """

    def add_chart_references(
        text: str,
        chart_data: Dict
    ) -> str:
        """
        Adds chart references to text
        
        Parameters:
            text: Input text
            chart_data: Chart data
            
        Returns:
            Text with chart references
        """
```

## Configuration

### LLM Settings
```python
LLM_CONFIG = {
    'openai': {
        'model': 'gpt-4',
        'temperature': 0.7,
        'max_tokens': 1000
    },
    'anthropic': {
        'model': 'claude-3-sonnet',
        'temperature': 0.7,
        'max_tokens': 1000
    }
}
```

### Context Settings
```python
CONTEXT_CONFIG = {
    'max_history': 10,
    'max_tokens': 4000,
    'token_buffer': 500
}
```

## Error Handling

### LLM Errors
```python
class LLMError(Exception):
    """Base class for LLM errors"""
    pass

class APIError(LLMError):
    """Raised when API call fails"""
    pass

class TokenLimitError(LLMError):
    """Raised when token limit exceeded"""
    pass
```

## Testing

### Unit Tests
1. LLM Service Tests
   - API integration
   - Error handling
   - Response parsing

2. Context Management Tests
   - Message history
   - Token counting
   - Context trimming

3. Function Calling Tests
   - Function registration
   - Parameter validation
   - Result formatting

### Integration Tests
1. End-to-end Query Tests
   - Query processing
   - Response generation
   - Context management

## Best Practices

### Prompt Engineering
1. Use clear and specific instructions
2. Provide relevant context
3. Handle edge cases
4. Maintain consistency

### Context Management
1. Monitor token usage
2. Preserve important context
3. Handle state transitions
4. Implement proper cleanup

### Function Calling
1. Validate parameters
2. Handle errors gracefully
3. Format results consistently
4. Document function behavior

## Future Improvements

### Planned Enhancements
1. Advanced prompt optimization
2. Improved context management
3. Enhanced function calling
4. Better error recovery

### Technical Debt
1. Optimize token usage
2. Improve error handling
3. Enhance documentation
4. Add performance metrics 