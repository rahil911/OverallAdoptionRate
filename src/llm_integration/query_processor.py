"""
Query Processor module for natural language query processing.

This module provides functionality for processing user queries:
- Intent recognition (descriptive, diagnostic, predictive, prescriptive)
- Entity extraction (dates, metrics, parameters)
- Query validation and relevance checking
- Query transformation to database parameters
- Context awareness for follow-up questions
- Query refinement suggestions
"""

import re
import datetime
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from dateutil.parser import parse as parse_date
from dateutil.relativedelta import relativedelta

# Import prompt templates for intent classification
from .prompts import determine_prompt_type

# Set up logging
logger = logging.getLogger(__name__)

# Define constants for entity types
class EntityType:
    """Types of entities that can be extracted from queries"""
    DATE = "date"
    METRIC = "metric"
    COMPARISON = "comparison"
    BUSINESS_UNIT = "business_unit"
    TIME_PERIOD = "time_period"
    ANOMALY = "anomaly"
    TREND = "trend"
    TARGET = "target"


class TimeFrame:
    """Common time frames for queries"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    

class QueryIntent:
    """Intent types for user queries"""
    DESCRIPTIVE = "descriptive"
    DIAGNOSTIC = "diagnostic"
    PREDICTIVE = "predictive"
    PRESCRIPTIVE = "prescriptive"
    UNKNOWN = "unknown"


class IntentClassifier:
    """Classifies user queries into intents"""
    
    def __init__(self):
        """Initialize the classifier with keyword patterns"""
        self.intent_patterns = {
            QueryIntent.DESCRIPTIVE: [
                r"what (is|was|are|were)",
                r"how (many|much)",
                r"show me",
                r"tell me about",
                r"describe",
                r"display",
                r"give me details",
                r"statistics",
                r"summary",
                r"overview",
                r"current"
            ],
            QueryIntent.DIAGNOSTIC: [
                r"why (is|was|are|were|did)",
                r"what caused",
                r"how come",
                r"reason",
                r"explain",
                r"understand why",
                r"analyze why",
                r"what factors",
                r"due to",
                r"because of",
                r"what happened",
                r"root cause"
            ],
            QueryIntent.PREDICTIVE: [
                r"will",
                r"forecast",
                r"predict",
                r"projection",
                r"future",
                r"trend",
                r"expect",
                r"anticipate",
                r"next",
                r"upcoming",
                r"going to",
                r"when will",
                r"how soon",
                r"estimate"
            ],
            QueryIntent.PRESCRIPTIVE: [
                r"how (can|could|should|would)",
                r"what (can|could|should|would)",
                r"recommend",
                r"suggest",
                r"advice",
                r"improve",
                r"increase",
                r"decrease",
                r"optimize",
                r"best way",
                r"steps to",
                r"action",
                r"strategy",
                r"plan"
            ]
        }
    
    def classify(self, query: str) -> str:
        """
        Classify a user query into an intent.
        
        Args:
            query: The user's query string
        
        Returns:
            str: The classified intent
        """
        # Normalize the query
        normalized_query = query.lower().strip()
        
        # Score each intent based on keyword matches
        intent_scores = {intent: 0 for intent in self.intent_patterns}
        
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, normalized_query, re.IGNORECASE):
                    intent_scores[intent] += 1
        
        # Get the intent with the highest score
        max_score = max(intent_scores.values())
        if max_score == 0:
            # No clear pattern match, use the fallback method
            return determine_prompt_type(query)
        
        # Find all intents with the max score
        top_intents = [intent for intent, score in intent_scores.items() if score == max_score]
        if len(top_intents) == 1:
            return top_intents[0]
        
        # If there's a tie, use the fallback method
        return determine_prompt_type(query)


class EntityExtractor:
    """Extracts entities from user queries"""
    
    def __init__(self):
        """Initialize the entity extractor with patterns"""
        # Date patterns
        self.date_patterns = [
            # ISO date: 2023-01-15
            (r'\b(\d{4}-\d{1,2}-\d{1,2})\b', self._parse_iso_date),
            # Month Year: January 2023, Jan 2023
            (r'\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{4})\b', self._parse_month_year),
            # Year-Month: 2023-01
            (r'\b(\d{4}-\d{1,2})\b', self._parse_year_month),
            # Year: 2023
            (r'\b(20\d{2})\b', self._parse_year),
            # Relative time: last month, past 3 weeks
            (r'\b(last|past|previous|next|upcoming|coming)\s+(\d+)?\s*(day|week|month|quarter|year)s?\b', self._parse_relative_time),
            # Quarter: Q1 2023
            (r'\bQ([1-4])\s+(\d{4})\b', self._parse_quarter)
        ]
        
        # Metric patterns
        self.metric_patterns = [
            # Adoption rate metrics
            (r'\b(adoption rate|overall adoption|adoption metrics?)\b', 'adoption_rate'),
            # DAU
            (r'\b(DAU|daily active users|daily adoption|daily usage)\b', 'daily_active_users'),
            # WAU
            (r'\b(WAU|weekly active users|weekly adoption|weekly usage)\b', 'weekly_active_users'),
            # MAU
            (r'\b(MAU|monthly active users|monthly adoption|monthly usage)\b', 'monthly_active_users'),
            # YAU
            (r'\b(YAU|yearly active users|yearly adoption|yearly usage)\b', 'yearly_active_users'),
        ]
        
        # Comparison patterns
        self.comparison_patterns = [
            # Month-over-Month
            (r'\b(MoM|month over month|month-over-month|monthly comparison)\b', 'month_over_month'),
            # Quarter-over-Quarter
            (r'\b(QoQ|quarter over quarter|quarter-over-quarter|quarterly comparison)\b', 'quarter_over_quarter'),
            # Year-over-Year
            (r'\b(YoY|year over year|year-over-year|yearly comparison|annual comparison)\b', 'year_over_year'),
            # Compare with
            (r'\bcompar(e|ing|ison)\s+(with|to|against)\s+(.+?)\b', 'custom_comparison'),
        ]
        
        # Time period patterns
        self.time_period_patterns = [
            # Daily
            (r'\b(day|daily|24 hours|per day)\b', TimeFrame.DAILY),
            # Weekly
            (r'\b(week|weekly|7 days|per week)\b', TimeFrame.WEEKLY),
            # Monthly
            (r'\b(month|monthly|30 days|per month)\b', TimeFrame.MONTHLY),
            # Quarterly
            (r'\b(quarter|quarterly|3 months|per quarter)\b', TimeFrame.QUARTERLY),
            # Yearly
            (r'\b(year|yearly|annual|annually|12 months|per year)\b', TimeFrame.YEARLY),
        ]
        
        # Anomaly patterns
        self.anomaly_patterns = [
            # Anomalies, outliers
            (r'\b(anomal(y|ies)|outlier|unusual|abnormal|unexpected|spike|drop|peak|valley)\b', 'anomaly'),
        ]
        
        # Trend patterns
        self.trend_patterns = [
            # Trends
            (r'\b(trend|pattern|movement|trajectory|direction|progress|course|path)\b', 'trend'),
        ]
        
        # Target patterns
        self.target_patterns = [
            # Numeric targets
            (r'\btarget\s+of\s+(\d+(\.\d+)?%?)\b', self._parse_target),
            # Increase/decrease targets
            (r'\b(increase|decrease|improve|reduce|boost|grow|decline)\s+by\s+(\d+(\.\d+)?%?)\b', self._parse_target),
            # Reach a level
            (r'\breach\s+(\d+(\.\d+)?%?)\b', self._parse_target),
        ]
    
    def extract_entities(self, query: str) -> Dict[str, List[Any]]:
        """
        Extract entities from a user query.
        
        Args:
            query: The user's query string
        
        Returns:
            Dict[str, List[Any]]: Dictionary of entity types and their values
        """
        # Normalize the query
        normalized_query = query.lower().strip()
        
        # Initialize the entities dictionary
        entities = {
            EntityType.DATE: [],
            EntityType.METRIC: [],
            EntityType.COMPARISON: [],
            EntityType.TIME_PERIOD: [],
            EntityType.ANOMALY: [],
            EntityType.TREND: [],
            EntityType.TARGET: []
        }
        
        # Extract dates
        for pattern, parser in self.date_patterns:
            for match in re.finditer(pattern, normalized_query, re.IGNORECASE):
                try:
                    date_entity = parser(match)
                    if date_entity:
                        entities[EntityType.DATE].append(date_entity)
                except Exception as e:
                    logger.warning(f"Error parsing date: {e}")
        
        # Extract metrics
        for pattern, metric_type in self.metric_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                entities[EntityType.METRIC].append(metric_type)
        
        # Extract comparisons
        for pattern, comparison_type in self.comparison_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                entities[EntityType.COMPARISON].append(comparison_type)
        
        # Extract time periods
        for pattern, period_type in self.time_period_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                entities[EntityType.TIME_PERIOD].append(period_type)
        
        # Extract anomalies
        for pattern, anomaly_type in self.anomaly_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                entities[EntityType.ANOMALY].append(anomaly_type)
        
        # Extract trends
        for pattern, trend_type in self.trend_patterns:
            if re.search(pattern, normalized_query, re.IGNORECASE):
                entities[EntityType.TREND].append(trend_type)
        
        # Extract targets
        for pattern, parser in self.target_patterns:
            for match in re.finditer(pattern, normalized_query, re.IGNORECASE):
                try:
                    target_entity = parser(match)
                    if target_entity:
                        entities[EntityType.TARGET].append(target_entity)
                except Exception as e:
                    logger.warning(f"Error parsing target: {e}")
        
        # Remove duplicate entities for simple types (strings)
        for entity_type in [EntityType.METRIC, EntityType.COMPARISON, EntityType.TIME_PERIOD, 
                            EntityType.ANOMALY, EntityType.TREND]:
            if entities[entity_type]:
                entities[entity_type] = list(set(entities[entity_type]))
        
        # For complex entity types like dates and targets, we can't use set()
        # because dictionaries are unhashable. Instead, we'll do a manual deduplication.
        for entity_type in [EntityType.DATE, EntityType.TARGET]:
            if entities[entity_type]:
                # Create a new deduplicated list
                deduplicated = []
                for entity in entities[entity_type]:
                    # Only add if not already in deduplicated list
                    if not any(self._dict_equals(entity, existing) for existing in deduplicated):
                        deduplicated.append(entity)
                
                entities[entity_type] = deduplicated
        
        return entities
    
    def _dict_equals(self, dict1: Dict, dict2: Dict) -> bool:
        """
        Compare two dictionaries for equality
        
        Args:
            dict1: First dictionary
            dict2: Second dictionary
            
        Returns:
            bool: True if dictionaries have the same keys and values
        """
        if set(dict1.keys()) != set(dict2.keys()):
            return False
            
        for key in dict1.keys():
            if dict1[key] != dict2[key]:
                return False
                
        return True
    
    def _parse_iso_date(self, match) -> Dict[str, Any]:
        """Parse ISO date format (YYYY-MM-DD)"""
        date_str = match.group(1)
        try:
            parsed_date = parse_date(date_str)
            return {
                'type': 'specific_date',
                'value': parsed_date.date(),
                'original': date_str
            }
        except:
            return None
    
    def _parse_month_year(self, match) -> Dict[str, Any]:
        """Parse month and year format (January 2023)"""
        month_str = match.group(1)
        year_str = match.group(2)
        
        # Map abbreviated months to full names
        month_map = {
            'Jan': 'January', 'Feb': 'February', 'Mar': 'March',
            'Apr': 'April', 'May': 'May', 'Jun': 'June',
            'Jul': 'July', 'Aug': 'August', 'Sep': 'September',
            'Oct': 'October', 'Nov': 'November', 'Dec': 'December'
        }
        
        if month_str in month_map:
            month_str = month_map[month_str]
        
        try:
            date_str = f"{month_str} 1, {year_str}"
            parsed_date = parse_date(date_str)
            
            # Get the last day of the month
            if parsed_date.month == 12:
                end_date = datetime.date(parsed_date.year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                end_date = datetime.date(parsed_date.year, parsed_date.month + 1, 1) - datetime.timedelta(days=1)
            
            return {
                'type': 'month_year',
                'start_date': parsed_date.date(),
                'end_date': end_date,
                'month': parsed_date.month,
                'year': parsed_date.year,
                'original': f"{month_str} {year_str}"
            }
        except:
            return None
    
    def _parse_year_month(self, match) -> Dict[str, Any]:
        """Parse year-month format (2023-01)"""
        date_str = match.group(1)
        try:
            # Append day 01 for parsing
            if len(date_str.split('-')) == 2:
                date_str += '-01'
            
            parsed_date = parse_date(date_str)
            
            # Get the last day of the month
            if parsed_date.month == 12:
                end_date = datetime.date(parsed_date.year + 1, 1, 1) - datetime.timedelta(days=1)
            else:
                end_date = datetime.date(parsed_date.year, parsed_date.month + 1, 1) - datetime.timedelta(days=1)
            
            return {
                'type': 'month_year',
                'start_date': parsed_date.date(),
                'end_date': end_date,
                'month': parsed_date.month,
                'year': parsed_date.year,
                'original': date_str.split('-01')[0]
            }
        except:
            return None
    
    def _parse_year(self, match) -> Dict[str, Any]:
        """Parse year format (2023)"""
        year_str = match.group(1)
        try:
            year = int(year_str)
            start_date = datetime.date(year, 1, 1)
            end_date = datetime.date(year, 12, 31)
            
            return {
                'type': 'year',
                'start_date': start_date,
                'end_date': end_date,
                'year': year,
                'original': year_str
            }
        except:
            return None
    
    def _parse_relative_time(self, match) -> Dict[str, Any]:
        """Parse relative time expressions (last month, past 3 weeks)"""
        direction = match.group(1)  # last, past, next, etc.
        count_str = match.group(2)  # number or None
        unit = match.group(3)  # day, week, month, etc.
        
        count = int(count_str) if count_str else 1
        is_future = direction in ['next', 'upcoming', 'coming']
        
        # Get current date
        today = datetime.date.today()
        
        if unit == 'day':
            delta = datetime.timedelta(days=count)
        elif unit == 'week':
            delta = datetime.timedelta(weeks=count)
        elif unit == 'month':
            delta = relativedelta(months=count)
        elif unit == 'quarter':
            delta = relativedelta(months=3 * count)
        elif unit == 'year':
            delta = relativedelta(years=count)
        else:
            return None
        
        if is_future:
            start_date = today
            end_date = today + delta
        else:
            start_date = today - delta
            end_date = today
        
        return {
            'type': 'relative',
            'start_date': start_date,
            'end_date': end_date,
            'direction': 'future' if is_future else 'past',
            'count': count,
            'unit': unit,
            'original': match.group(0)
        }
    
    def _parse_quarter(self, match) -> Dict[str, Any]:
        """Parse quarter format (Q1 2023)"""
        quarter = int(match.group(1))
        year = int(match.group(2))
        
        if quarter < 1 or quarter > 4:
            return None
        
        start_month = (quarter - 1) * 3 + 1
        start_date = datetime.date(year, start_month, 1)
        
        if quarter == 4:
            end_date = datetime.date(year + 1, 1, 1) - datetime.timedelta(days=1)
        else:
            end_date = datetime.date(year, start_month + 3, 1) - datetime.timedelta(days=1)
        
        return {
            'type': 'quarter',
            'start_date': start_date,
            'end_date': end_date,
            'quarter': quarter,
            'year': year,
            'original': match.group(0)
        }
    
    def _parse_target(self, match) -> Dict[str, Any]:
        """Parse target values"""
        full_match = match.group(0)
        
        # Try to extract numeric value
        value_match = re.search(r'(\d+(\.\d+)?)', full_match)
        if not value_match:
            return None
        
        value = float(value_match.group(1))
        
        # Check if it's a percentage
        is_percentage = '%' in full_match
        
        # Determine action (increase, decrease, target)
        if 'increase' in full_match or 'improve' in full_match or 'boost' in full_match or 'grow' in full_match:
            action = 'increase'
        elif 'decrease' in full_match or 'reduce' in full_match or 'decline' in full_match:
            action = 'decrease'
        else:
            action = 'target'
        
        return {
            'type': 'target',
            'value': value,
            'is_percentage': is_percentage,
            'action': action,
            'original': full_match
        }


class QueryValidator:
    """Validates user queries for relevance and completeness"""
    
    def __init__(self):
        """Initialize the query validator"""
        # Keywords related to the domain
        self.domain_keywords = [
            'adoption', 'rate', 'user', 'active', 'dau', 'wau', 'mau', 'yau',
            'daily', 'weekly', 'monthly', 'yearly', 'trend', 'increase', 'decrease',
            'metric', 'measure', 'opus', 'product', 'platform', 'performance', 'growth'
        ]
        
        # Required entity combinations for different intents
        self.required_entities = {
            QueryIntent.DESCRIPTIVE: [EntityType.METRIC, EntityType.DATE],
            QueryIntent.DIAGNOSTIC: [EntityType.METRIC, EntityType.DATE],
            QueryIntent.PREDICTIVE: [EntityType.METRIC],
            QueryIntent.PRESCRIPTIVE: [EntityType.METRIC]
        }
    
    def validate_domain_relevance(self, query: str) -> bool:
        """
        Check if the query is relevant to the adoption rate domain.
        
        Args:
            query: The user's query string
        
        Returns:
            bool: True if relevant, False otherwise
        """
        normalized_query = query.lower()
        # Count how many domain keywords are in the query
        keyword_count = sum(1 for keyword in self.domain_keywords if keyword in normalized_query)
        
        # If at least 2 domain keywords are present, consider it relevant
        return keyword_count >= 1
    
    def validate_entity_completeness(self, intent: str, entities: Dict[str, List[Any]]) -> Tuple[bool, Optional[str]]:
        """
        Check if the query has all required entities for the intent.
        
        Args:
            intent: The classified intent
            entities: Extracted entities from the query
        
        Returns:
            Tuple[bool, Optional[str]]: (is_complete, missing_entity_message)
        """
        if intent not in self.required_entities:
            return True, None
        
        missing_entities = []
        
        # Check required entities for the intent
        for required_entity in self.required_entities[intent]:
            if not entities.get(required_entity, []):
                missing_entities.append(required_entity)
        
        if missing_entities:
            missing_message = f"Missing information: {', '.join(missing_entities)}"
            return False, missing_message
        
        return True, None
    
    def generate_query_refinement(self, intent: str, entities: Dict[str, List[Any]]) -> Optional[str]:
        """
        Generate refinement suggestions for incomplete queries.
        
        Args:
            intent: The classified intent
            entities: Extracted entities from the query
        
        Returns:
            Optional[str]: Refinement suggestion or None if not needed
        """
        is_complete, missing_message = self.validate_entity_completeness(intent, entities)
        
        if not is_complete:
            refinements = []
            
            if EntityType.DATE not in entities or not entities[EntityType.DATE]:
                refinements.append("Please specify a time period (e.g., 'March 2024', 'last month', 'Q1 2023')")
            
            if EntityType.METRIC not in entities or not entities[EntityType.METRIC]:
                refinements.append("Please specify which metric you're interested in (e.g., 'adoption rate', 'DAU', 'MAU')")
            
            # For predictive queries without a future reference
            if intent == QueryIntent.PREDICTIVE and not any('future' in str(e) for e in entities.get(EntityType.DATE, [])):
                refinements.append("Please specify the future period for prediction (e.g., 'next quarter', 'by end of 2024')")
            
            if refinements:
                return "To refine your query:\n" + "\n".join(f"- {r}" for r in refinements)
        
        return None


class QueryTransformer:
    """Transforms user queries into database query parameters"""
    
    def __init__(self):
        """Initialize the query transformer"""
        # Default values
        self.default_tenant_id = 1388
        self.default_lookback_days = 90  # For queries without specific dates
    
    def transform_to_database_params(
        self, 
        intent: str, 
        entities: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """
        Transform entities to database query parameters.
        
        Args:
            intent: The classified intent
            entities: Extracted entities from the query
        
        Returns:
            Dict[str, Any]: Parameters ready for database queries
        """
        params = {
            'tenant_id': self.default_tenant_id
        }
        
        # Process date entities
        if entities.get(EntityType.DATE):
            date_entity = self._select_best_date_entity(entities[EntityType.DATE])
            if date_entity:
                if date_entity.get('type') == 'specific_date':
                    params['from_date'] = date_entity['value']
                    params['to_date'] = date_entity['value']
                else:
                    params['from_date'] = date_entity.get('start_date')
                    params['to_date'] = date_entity.get('end_date')
        else:
            # Default date range if none specified
            today = datetime.date.today()
            params['from_date'] = today - datetime.timedelta(days=self.default_lookback_days)
            params['to_date'] = today
        
        # Process metric entities
        if entities.get(EntityType.METRIC):
            metric = entities[EntityType.METRIC][0]
            params['metric_type'] = self._map_metric_to_db_param(metric)
        else:
            # Default to monthly adoption rate if none specified
            params['metric_type'] = 'monthly'
        
        # Process time period entities (for aggregation level)
        if entities.get(EntityType.TIME_PERIOD):
            time_period = entities[EntityType.TIME_PERIOD][0]
            params['aggregation'] = time_period
        
        # Process comparison entities
        if entities.get(EntityType.COMPARISON):
            comparison = entities[EntityType.COMPARISON][0]
            params['comparison_type'] = comparison
            
            # For custom comparisons, we need to determine the comparison period
            if comparison == 'custom_comparison' and len(entities.get(EntityType.DATE, [])) >= 2:
                date_entities = self._sort_date_entities(entities[EntityType.DATE])
                if len(date_entities) >= 2:
                    params['period1_start'] = date_entities[0].get('start_date')
                    params['period1_end'] = date_entities[0].get('end_date')
                    params['period2_start'] = date_entities[1].get('start_date')
                    params['period2_end'] = date_entities[1].get('end_date')
        
        # Process anomaly entities
        if entities.get(EntityType.ANOMALY):
            params['analyze_anomalies'] = True
            params['anomaly_method'] = 'ensemble'  # Default to ensemble method
        
        # Process trend entities
        if entities.get(EntityType.TREND):
            params['analyze_trend'] = True
        
        # Process target entities
        if entities.get(EntityType.TARGET):
            target = entities[EntityType.TARGET][0]
            params['target_value'] = target.get('value')
            params['target_is_percentage'] = target.get('is_percentage', False)
            params['target_action'] = target.get('action')
        
        return params
    
    def _select_best_date_entity(self, date_entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Select the most appropriate date entity from multiple candidates"""
        if not date_entities:
            return None
        
        # Prioritize entities with both start and end dates
        complete_entities = [e for e in date_entities if 'start_date' in e and 'end_date' in e]
        if complete_entities:
            # Prefer the most recent time period
            return max(complete_entities, key=lambda e: e['end_date'])
        
        # If no complete entities, just return the first one
        return date_entities[0]
    
    def _sort_date_entities(self, date_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Sort date entities by start date"""
        complete_entities = [e for e in date_entities if 'start_date' in e]
        return sorted(complete_entities, key=lambda e: e['start_date'])
    
    def _map_metric_to_db_param(self, metric: str) -> str:
        """Map a metric entity to a database parameter"""
        metric_map = {
            'adoption_rate': 'monthly',
            'daily_active_users': 'daily',
            'weekly_active_users': 'weekly',
            'monthly_active_users': 'monthly',
            'yearly_active_users': 'yearly'
        }
        
        return metric_map.get(metric, 'monthly')


class ContextTracker:
    """Tracks conversation context for follow-up questions"""
    
    def __init__(self):
        """Initialize the context tracker"""
        self.current_context = {
            'intent': None,
            'entities': {},
            'params': {},
            'last_query': None,
            'follow_up_count': 0,
            'topics_discussed': set()
        }
    
    def update_context(
        self, 
        query: str, 
        intent: str, 
        entities: Dict[str, List[Any]], 
        params: Dict[str, Any]
    ) -> None:
        """
        Update the conversation context with new information.
        
        Args:
            query: The user's query string
            intent: The classified intent
            entities: Extracted entities
            params: Database query parameters
        """
        # Check if this is a follow-up question
        is_follow_up = self._is_follow_up_question(query)
        
        if is_follow_up:
            # Increment follow-up count
            self.current_context['follow_up_count'] += 1
            
            # Merge new entities with existing ones, prioritizing new entities
            for entity_type, entity_values in entities.items():
                if entity_values:  # Only update if new entities were found
                    self.current_context['entities'][entity_type] = entity_values
            
            # Merge params, prioritizing new params
            for param, value in params.items():
                if value is not None:  # Only update if new value is provided
                    self.current_context['params'][param] = value
        else:
            # New topic, reset context
            self.current_context = {
                'intent': intent,
                'entities': entities,
                'params': params,
                'last_query': query,
                'follow_up_count': 0,
                'topics_discussed': self.current_context.get('topics_discussed', set())
            }
            
            # Add current intent to topics discussed
            self.current_context['topics_discussed'].add(intent)
    
    def get_current_context(self) -> Dict[str, Any]:
        """
        Get the current conversation context.
        
        Returns:
            Dict[str, Any]: Current context state
        """
        return self.current_context
    
    def _is_follow_up_question(self, query: str) -> bool:
        """
        Determine if a query is a follow-up question.
        
        Args:
            query: The user's query string
        
        Returns:
            bool: True if it's a follow-up question
        """
        # If no previous query, it's not a follow-up
        if not self.current_context.get('last_query'):
            return False
        
        # Look for follow-up indicators
        follow_up_indicators = [
            # Pronouns referring to previous content
            r'\b(it|this|that|these|those|they|them)\b',
            # Questions starting with "and", "but", "also"
            r'^\s*(and|but|also|what about|how about)',
            # Questions starting without a subject
            r'^\s*(what|how|why|when|where|who|which)\s+(about|if|is|was|are|were)\b',
            # Ellipsis continuing from previous
            r'^\s*\.{2,}',
            # Very short questions (likely follow-ups)
            r'^\s*([^.?!]{1,25})[.?!]?\s*$'
        ]
        
        # Check if any indicator matches
        for indicator in follow_up_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                return True
        
        return False
    
    def generate_clarification_question(self) -> Optional[str]:
        """
        Generate a clarification question based on the current context.
        
        Returns:
            Optional[str]: Clarification question or None if not needed
        """
        if not self.current_context.get('intent'):
            return None
        
        intent = self.current_context['intent']
        entities = self.current_context.get('entities', {})
        
        missing_entities = []
        
        # Check for missing crucial entities based on intent
        if intent == QueryIntent.DESCRIPTIVE or intent == QueryIntent.DIAGNOSTIC:
            if not entities.get(EntityType.DATE):
                missing_entities.append("time period")
            if not entities.get(EntityType.METRIC):
                missing_entities.append("specific metric")
        
        elif intent == QueryIntent.PREDICTIVE:
            # For predictions, we need future time reference
            future_dates = [e for e in entities.get(EntityType.DATE, []) 
                           if e.get('direction') == 'future']
            if not future_dates:
                missing_entities.append("future time period")
        
        elif intent == QueryIntent.PRESCRIPTIVE:
            # For recommendations, we need a target
            if not entities.get(EntityType.TARGET):
                missing_entities.append("target or goal")
        
        if missing_entities:
            return f"Could you specify the {' and '.join(missing_entities)} you're interested in?"
        
        return None


class QueryProcessor:
    """Main class for natural language query processing"""
    
    def __init__(self):
        """Initialize the query processor with all components"""
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.query_validator = QueryValidator()
        self.query_transformer = QueryTransformer()
        self.context_tracker = ContextTracker()
    
    def process_query(self, query: str, use_context: bool = True) -> Dict[str, Any]:
        """
        Process a natural language query.
        
        Args:
            query: The user's query string
            use_context: Whether to use conversation context
        
        Returns:
            Dict[str, Any]: Processing results with intent, entities, 
                          database parameters, and validation info
        """
        # Step 1: Classify intent
        intent = self.intent_classifier.classify(query)
        
        # Step 2: Extract entities
        entities = self.entity_extractor.extract_entities(query)
        
        # Step 3: Validate query
        is_relevant = self.query_validator.validate_domain_relevance(query)
        is_complete, missing_message = self.query_validator.validate_entity_completeness(intent, entities)
        refinement_suggestion = None
        
        if not is_relevant:
            refinement_suggestion = "Your question doesn't seem to be related to adoption rates. Please ask about adoption metrics, trends, or user activity."
        elif not is_complete:
            refinement_suggestion = self.query_validator.generate_query_refinement(intent, entities)
        
        # Step 4: Transform to database parameters
        current_context = self.context_tracker.get_current_context() if use_context else {}
        
        # If this is a follow-up, merge with current context
        if use_context and self._is_follow_up_question(query, current_context):
            # Merge entities from context for missing types
            for entity_type, entity_values in current_context.get('entities', {}).items():
                if not entities.get(entity_type):
                    entities[entity_type] = entity_values
            
            # Use the intent from context if none was clearly identified
            if intent == QueryIntent.UNKNOWN and current_context.get('intent'):
                intent = current_context['intent']
        
        # Transform to database parameters
        params = self.query_transformer.transform_to_database_params(intent, entities)
        
        # Step 5: Update context
        if use_context:
            self.context_tracker.update_context(query, intent, entities, params)
        
        # Step 6: Generate clarification if needed
        clarification_question = None
        if not is_complete and use_context:
            clarification_question = self.context_tracker.generate_clarification_question()
        
        # Prepare result
        result = {
            'query': query,
            'intent': intent,
            'entities': entities,
            'params': params,
            'is_relevant': is_relevant,
            'is_complete': is_complete,
            'refinement_suggestion': refinement_suggestion,
            'clarification_question': clarification_question,
            'is_follow_up': self._is_follow_up_question(query, current_context) if use_context else False
        }
        
        return result
    
    def _is_follow_up_question(self, query: str, context: Dict[str, Any]) -> bool:
        """Check if a query is a follow-up question based on context"""
        if not context.get('last_query'):
            return False
        
        # Look for follow-up indicators
        follow_up_indicators = [
            # Pronouns referring to previous content
            r'\b(it|this|that|these|those|they|them)\b',
            # Questions starting with "and", "but", "also"
            r'^\s*(and|but|also|what about|how about)',
            # Questions without clear subject
            r'^\s*(what|how|why|when|where|who|which)\s+(about|if|is|was|are|were)\b',
            # Very short questions (likely follow-ups)
            r'^\s*([^.?!]{1,25})[.?!]?\s*$'
        ]
        
        # Check if any indicator matches
        for indicator in follow_up_indicators:
            if re.search(indicator, query, re.IGNORECASE):
                return True
        
        return False 