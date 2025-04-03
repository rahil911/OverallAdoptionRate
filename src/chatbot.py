"""
Overall Adoption Rate Chatbot

This module integrates the query processor with the LLM service to process natural
language queries about adoption rate data and generates informative responses.
"""

import os
import logging
import datetime
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
import json

# Import components from our application
from src.database.data_access import DataAccessLayer
from src.data_processing.data_fetcher import DataFetcher
from src.data_analysis.trend_analyzer import generate_trend_description
from src.data_analysis.anomaly_detector import detect_anomalies_ensemble
from src.llm_integration.query_processor import QueryProcessor
from src.llm_integration.llm_service import generate_openai_response, generate_anthropic_response
from src.llm_integration.context_manager import MessageHistory
from src.llm_integration.prompts import (
    get_system_prompt, 
    get_descriptive_prompt_template,
    get_diagnostic_prompt_template,
    get_predictive_prompt_template,
    get_prescriptive_prompt_template
)
from src.llm_integration.function_calling import (
    get_function_definitions,
    handle_function_response,
    execute_function
)
from src.llm_integration.response_formatter import format_response

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class AdoptionRateChatbot:
    """Main chatbot class for handling adoption rate queries"""
    
    def __init__(self, use_anthropic: bool = False):
        """
        Initialize the chatbot with components
        
        Args:
            use_anthropic: Whether to use Anthropic instead of OpenAI
        """
        # Set up components
        self.query_processor = QueryProcessor()
        self.data_fetcher = DataFetcher(DataAccessLayer)
        self.tenant_id = 1388  # Default tenant for testing
        self.use_anthropic = use_anthropic
        
        # Set up message history with system prompt
        system_prompt = """
You are an adoption rate specialist assistant helping users understand their Opus product adoption metrics.
Your goal is to provide clear, concise insights based on actual adoption rate data.
When providing information, focus on key insights and trends, and use precise numerical data.
Always refer to adoption rate as a percentage between 0-100%.
If asked about improving adoption rates, provide specific, actionable recommendations based on the data patterns.
"""
        self.message_history = MessageHistory(system_prompt=system_prompt)
        
        logger.info("Adoption Rate Chatbot initialized")
    
    def process_query(self, user_query: str) -> str:
        """
        Process a user query and return a response.
        
        Args:
            user_query: The user's natural language query
            
        Returns:
            str: The generated response
        """
        logger.info(f"Processing query: {user_query}")
        
        try:
            # Add user message to history
            self.message_history.add_message("user", user_query)
            
            # Process the query with NLP
            query_result = self.query_processor.process_query(user_query)
            
            # Check if the query is relevant to the domain
            if not query_result['is_relevant']:
                response = self._handle_irrelevant_query(query_result)
                self.message_history.add_message("assistant", response)
                return response
            
            # Check if the query is complete
            if not query_result['is_complete']:
                response = self._handle_incomplete_query(query_result)
                self.message_history.add_message("assistant", response)
                return response
            
            # Generate prompt based on intent
            prompt_template = self._get_prompt_template(query_result['intent'])
            
            # Get data needed for the query
            prompt_data = self._get_data_for_query(query_result)
            
            # Format the prompt with data
            formatted_prompt = prompt_template.format(**prompt_data)
            
            # Log the prompt being sent to verify it's working properly
            logger.info(f"Sending prompt to LLM: {formatted_prompt[:200]}...")
            
            # Verify API key before making the API call
            api_key = None
            if self.use_anthropic:
                api_key = os.getenv('ANTHROPIC_API_KEY')
                logger.info(f"Using Anthropic API with key {'valid' if api_key else 'missing'}")
            else:
                api_key = os.getenv('OPENAI_API_KEY')
                logger.info(f"Using OpenAI API with key {'valid' if api_key else 'missing'}")
            
            if not api_key:
                logger.error(f"API key is missing or invalid for {'Anthropic' if self.use_anthropic else 'OpenAI'}")
                return "Error: API key is missing. Please check your environment variables."
            
            # Generate response using LLM
            if self.use_anthropic:
                logger.info("Calling Anthropic API for response")
                llm_response = generate_anthropic_response(
                    messages=self.message_history.get_messages() + [{"role": "user", "content": formatted_prompt}]
                )
            else:
                # Define functions for function calling
                functions = get_function_definitions()
                logger.info("Calling OpenAI API for response with function calling")
                
                llm_response = generate_openai_response(
                    messages=self.message_history.get_messages() + [{"role": "user", "content": formatted_prompt}],
                    functions=functions
                )
            
            # Log successful API response
            logger.info(f"Received response from LLM API")
            
            # Check if the response includes a tool/function call
            has_tool_call = False
            function_name = None
            function_args = '{}'
            
            # Extract tool call information based on response format
            if hasattr(llm_response, 'choices') and llm_response.choices:
                message = llm_response.choices[0].message
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    has_tool_call = True
                    tool_call = message.tool_calls[0]
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
            
            if has_tool_call:
                # Add function call to message history
                # Convert OpenAI response to a dictionary for storage
                response_dict = {
                    "role": "assistant",
                    "content": ""  # Ensure content is a string, not None
                }
                self.message_history.add_message("assistant", response_dict)
                
                # Prepare function call for execution
                function_call_obj = {
                    "name": function_name,
                    "arguments": json.loads(function_args) if isinstance(function_args, str) else function_args
                }
                
                # Execute the function
                function_result = execute_function(function_call_obj)
                
                # Format the function response
                function_response = handle_function_response(function_result, function_name)
                
                # Add function response to message history
                self.message_history.add_message("function", function_response, name=function_name)
                
                # Generate final response with function result
                final_response = self._generate_final_response(function_result, query_result)
                
                # Add final response to message history
                self.message_history.add_message("assistant", final_response)
                
                # Format the response before returning
                return format_response(final_response)
            else:
                # No function call, extract the direct response content
                response_text = ""
                if hasattr(llm_response, 'choices') and llm_response.choices:
                    message = llm_response.choices[0].message
                    response_text = message.content if message.content else ""
                
                # Add response to message history
                self.message_history.add_message("assistant", response_text)
                
                # Format the response before returning
                return format_response(response_text)
                
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return f"Error processing query: {str(e)}"
    
    def _handle_irrelevant_query(self, query_result: Dict[str, Any]) -> str:
        """Handle queries that are not relevant to adoption rate data"""
        return (
            "I'm an adoption rate specialist focused on helping you understand the adoption metrics "
            "for the Opus product. I don't have information about other topics. "
            "Could you please ask me something related to adoption rates, active users, or usage metrics?"
        )
    
    def _handle_incomplete_query(self, query_result: Dict[str, Any]) -> str:
        """Handle queries that are missing required information"""
        if query_result['refinement_suggestion']:
            return (
                "I need a bit more information to answer your question accurately.\n\n"
                f"{query_result['refinement_suggestion']}\n\n"
                "Could you please provide more details so I can give you a better response?"
            )
        else:
            return (
                "I need more specific information to answer your question. "
                "Please include details such as the time period (e.g., 'March 2024', 'Q1 2023') "
                "and the specific metric you're interested in (adoption rate, DAU, MAU, etc.)."
            )
    
    def _get_prompt_template(self, intent: str) -> str:
        """Get the appropriate prompt template based on intent"""
        # Return a simplified template that can be formatted later
        if intent == "diagnostic":
            return """
Based on the adoption rate data for {date_range}, analyze {metric_type} adoption rates.
Explain any patterns, trends, or anomalies visible in the data.
Provide specific numerical insights when available.

Data summary:
- Number of records: {num_records}
- Date range: {date_range}
- Average adoption rate: {avg_adoption_rate}%
- Min adoption rate: {min_adoption_rate}%
- Max adoption rate: {max_adoption_rate}%

User query: {user_query}
"""
        elif intent == "predictive":
            return """
Based on historical adoption rate data up to {query_date}, predict what we might expect to see
in terms of {metric_type} adoption rates for the future.
Support your prediction with trend analysis, seasonality patterns, and growth rates observed in the data.
Be clear about the confidence level of your prediction and what factors might cause deviations.

Historical data summary:
- Number of records: {num_records}
- Date range: {date_range}
- Average adoption rate: {avg_adoption_rate}%
- Min adoption rate: {min_adoption_rate}%
- Max adoption rate: {max_adoption_rate}%

User query: {user_query}
"""
        elif intent == "prescriptive":
            return """
Based on the adoption rate analysis for {date_range}, provide recommendations for improving {metric_type} adoption rates.
Your recommendations should be specific, actionable, and directly tied to insights from the data.
Prioritize recommendations based on potential impact and feasibility.

Data analysis summary:
- Number of records: {num_records}
- Date range: {date_range}
- Average adoption rate: {avg_adoption_rate}%
- Min adoption rate: {min_adoption_rate}%
- Max adoption rate: {max_adoption_rate}%

User query: {user_query}
"""
        else:  # Default to descriptive
            return """
Analyze the {metric_type} adoption rate data for {date_range}.
Provide a clear and concise description of the data, highlighting key metrics and trends.
Include specific numerical values and comparisons where relevant.

Data summary:
- Number of records: {num_records}
- Date range: {date_range}
- Average adoption rate: {avg_adoption_rate}%
- Min adoption rate: {min_adoption_rate}%
- Max adoption rate: {max_adoption_rate}%

User query: {user_query}
"""
    
    def _get_data_for_query(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve data needed for the query based on processed parameters
        
        Args:
            query_result: The processed query result with parameters
            
        Returns:
            Dict[str, Any]: Data to inject into the prompt template
        """
        params = query_result['params']
        from_date = params.get('from_date')
        to_date = params.get('to_date')
        metric_type = params.get('metric_type', 'monthly')
        
        # Get the adoption rate data
        adoption_data = self.data_fetcher.get_overall_adoption_rate(
            from_date=from_date,
            to_date=to_date,
            tenant_id=self.tenant_id
        )
        
        # Get the active users data
        if metric_type == 'daily':
            active_users = self.data_fetcher.get_daily_active_users(
                from_date=from_date,
                to_date=to_date,
                tenant_id=self.tenant_id
            )
        else:
            active_users = self.data_fetcher.get_monthly_active_users(
                from_date=from_date,
                to_date=to_date,
                tenant_id=self.tenant_id
            )
        
        # Convert model objects to dictionaries for easy access
        adoption_data_dicts = [item.to_dict() for item in adoption_data]
        active_users_dicts = [item.to_dict() for item in active_users]
        
        # Calculate statistics from the adoption data
        adoption_rate_key = f"{metric_type[0].lower()}_adoption_rate"  # e.g., "daily_adoption_rate" or "monthly_adoption_rate"
        
        # Extract adoption rates from objects
        adoption_rates = []
        for item in adoption_data:
            if metric_type == 'daily':
                adoption_rates.append(item.daily_adoption_rate)
            elif metric_type == 'weekly':
                adoption_rates.append(item.weekly_adoption_rate)
            elif metric_type == 'monthly':
                adoption_rates.append(item.monthly_adoption_rate)
            elif metric_type == 'yearly':
                adoption_rates.append(item.yearly_adoption_rate)
        
        num_records = len(adoption_rates)
        avg_adoption_rate = sum(adoption_rates) / num_records if num_records > 0 else 0
        min_adoption_rate = min(adoption_rates) if num_records > 0 else 0
        max_adoption_rate = max(adoption_rates) if num_records > 0 else 0
        
        # Determine date range string
        date_range_str = f"{from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}" if from_date and to_date else "all available data"
        
        # Compile basic data for all query types
        data = {
            "user_query": query_result['query'],
            "query_date": datetime.datetime.now().strftime("%Y-%m-%d"),
            "date_range": date_range_str,
            "adoption_data": adoption_data_dicts,
            "active_users": active_users_dicts,
            "metric_type": metric_type,
            "num_records": num_records,
            "avg_adoption_rate": round(avg_adoption_rate, 2),
            "min_adoption_rate": round(min_adoption_rate, 2),
            "max_adoption_rate": round(max_adoption_rate, 2)
        }
        
        # Add trend analysis if requested
        if params.get('analyze_trend', False):
            trend_description = generate_trend_description(adoption_data)
            data['trend_analysis'] = trend_description
        
        # Add anomaly analysis if requested
        if params.get('analyze_anomalies', False):
            anomaly_method = params.get('anomaly_method', 'ensemble')
            anomalies = detect_anomalies_ensemble(adoption_data)
            data['anomaly_analysis'] = anomalies
        
        return data
    
    def _generate_final_response(self, function_result: Any, query_result: Dict[str, Any]) -> str:
        """
        Generate a final response based on function result and query context
        
        Args:
            function_result: The result from executing the function
            query_result: The processed query result
            
        Returns:
            str: The final response text
        """
        # Set up a new prompt for final response
        intent = query_result['intent']
        prompt_template = self._get_prompt_template(intent)
        
        # Extract metrics from function result
        metric_type = query_result['params'].get('metric_type', 'monthly')
        
        # Safety checks for function result format and convert if needed
        processed_results = []
        if isinstance(function_result, list) and len(function_result) > 0:
            # Check if we have model objects or dictionaries
            if hasattr(function_result[0], 'to_dict'):
                # We have model objects
                processed_results = [item.to_dict() for item in function_result]
            else:
                # We already have dictionaries
                processed_results = function_result
            
            # Map attribute names based on metric type
            if metric_type == 'daily':
                adoption_rate_key = 'daily_adoption_rate'
            elif metric_type == 'weekly':
                adoption_rate_key = 'weekly_adoption_rate'
            elif metric_type == 'monthly':
                adoption_rate_key = 'monthly_adoption_rate'
            else:  # yearly
                adoption_rate_key = 'yearly_adoption_rate'
                
            # Extract adoption rates from the result
            adoption_rates = []
            for row in processed_results:
                if adoption_rate_key in row:
                    try:
                        rate = float(row[adoption_rate_key])
                        adoption_rates.append(rate)
                    except (ValueError, TypeError):
                        pass
            
            # Calculate statistics
            num_records = len(adoption_rates)
            avg_adoption_rate = sum(adoption_rates) / num_records if num_records > 0 else 0
            min_adoption_rate = min(adoption_rates) if num_records > 0 else 0
            max_adoption_rate = max(adoption_rates) if num_records > 0 else 0
            
            # Determine date range
            from_date = None
            to_date = None
            if num_records > 0:
                try:
                    if 'date' in processed_results[0]:
                        dates = []
                        for row in processed_results:
                            if 'date' in row:
                                try:
                                    # Date could be in ISO format or datetime object
                                    if isinstance(row['date'], str):
                                        dates.append(datetime.datetime.fromisoformat(row['date']).date())
                                    elif isinstance(row['date'], (datetime.date, datetime.datetime)):
                                        dates.append(row['date'].date() if isinstance(row['date'], datetime.datetime) else row['date'])
                                except (ValueError, TypeError):
                                    pass
                        
                        from_date = min(dates) if dates else None
                        to_date = max(dates) if dates else None
                except Exception as e:
                    logger.error(f"Error parsing dates: {e}")
            
            date_range = (f"{from_date.strftime('%Y-%m-%d')} to {to_date.strftime('%Y-%m-%d')}" 
                         if from_date and to_date else "provided time period")
            
            # Create context data with function result statistics
            context_data = {
                "user_query": query_result['query'],
                "query_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "date_range": date_range,
                "metric_type": metric_type,
                "num_records": num_records,
                "avg_adoption_rate": round(avg_adoption_rate, 2),
                "min_adoption_rate": round(min_adoption_rate, 2),
                "max_adoption_rate": round(max_adoption_rate, 2)
            }
            
            # Format the template with our data
            final_prompt = prompt_template.format(**context_data)
        else:
            # Fallback if function result isn't in expected format
            final_prompt = f"""
Based on the user query: "{query_result['query']}"
And the retrieved data: {function_result}

Please provide a comprehensive answer that addresses the user's question.
"""
        
        # Generate final response
        if self.use_anthropic:
            response = generate_anthropic_response(
                messages=self.message_history.get_messages() + [{"role": "user", "content": final_prompt}]
            )
        else:
            response = generate_openai_response(
                messages=self.message_history.get_messages() + [{"role": "user", "content": final_prompt}]
            )
        
        # Extract response text from OpenAI response object
        response_text = ""
        if hasattr(response, 'choices') and response.choices:
            message = response.choices[0].message
            response_text = message.content if message.content else ""
        elif isinstance(response, dict):
            response_text = response.get('content', '')
        elif isinstance(response, str):
            response_text = response
        
        return response_text


def create_chatbot(use_anthropic: bool = False) -> AdoptionRateChatbot:
    """
    Factory function to create a chatbot instance
    
    Args:
        use_anthropic: Whether to use Anthropic instead of OpenAI
        
    Returns:
        AdoptionRateChatbot: An initialized chatbot instance
    """
    # Make sure we have the needed environment variables
    required_keys = ['OPENAI_API_KEY'] if not use_anthropic else ['ANTHROPIC_API_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_keys)}")
    
    return AdoptionRateChatbot(use_anthropic=use_anthropic) 