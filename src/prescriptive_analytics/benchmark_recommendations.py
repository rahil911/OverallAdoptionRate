"""
Benchmark Recommendations Module

This module provides functionality for comparing adoption rates to industry benchmarks and
generating recommendations based on how the current rates measure up to best practices.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union, Tuple

# Set up logging
logger = logging.getLogger(__name__)

# Define industry benchmarks
INDUSTRY_BENCHMARKS = {
    "saas_overall": {
        "description": "Overall SaaS industry",
        "daily_adoption_rate": {
            "excellent": 25.0,
            "good": 15.0,
            "average": 10.0,
            "below_average": 5.0,
            "poor": 2.0
        },
        "weekly_adoption_rate": {
            "excellent": 40.0,
            "good": 30.0,
            "average": 20.0,
            "below_average": 10.0,
            "poor": 5.0
        },
        "monthly_adoption_rate": {
            "excellent": 60.0,
            "good": 45.0,
            "average": 30.0,
            "below_average": 15.0,
            "poor": 7.5
        },
        "yearly_adoption_rate": {
            "excellent": 80.0,
            "good": 65.0,
            "average": 50.0,
            "below_average": 30.0,
            "poor": 15.0
        }
    },
    "enterprise_software": {
        "description": "Enterprise software solutions",
        "daily_adoption_rate": {
            "excellent": 20.0,
            "good": 12.0,
            "average": 8.0,
            "below_average": 4.0,
            "poor": 1.5
        },
        "weekly_adoption_rate": {
            "excellent": 35.0,
            "good": 25.0,
            "average": 15.0,
            "below_average": 8.0,
            "poor": 4.0
        },
        "monthly_adoption_rate": {
            "excellent": 50.0,
            "good": 40.0,
            "average": 25.0,
            "below_average": 12.0,
            "poor": 6.0
        },
        "yearly_adoption_rate": {
            "excellent": 75.0,
            "good": 60.0,
            "average": 40.0,
            "below_average": 25.0,
            "poor": 12.0
        }
    },
    "productivity_tools": {
        "description": "Productivity and collaboration tools",
        "daily_adoption_rate": {
            "excellent": 30.0,
            "good": 20.0,
            "average": 12.0,
            "below_average": 6.0,
            "poor": 3.0
        },
        "weekly_adoption_rate": {
            "excellent": 50.0,
            "good": 35.0,
            "average": 25.0,
            "below_average": 12.0,
            "poor": 6.0
        },
        "monthly_adoption_rate": {
            "excellent": 70.0,
            "good": 55.0,
            "average": 40.0,
            "below_average": 20.0,
            "poor": 10.0
        },
        "yearly_adoption_rate": {
            "excellent": 85.0,
            "good": 75.0,
            "average": 60.0,
            "below_average": 35.0,
            "poor": 20.0
        }
    },
    "financial_software": {
        "description": "Financial and accounting software",
        "daily_adoption_rate": {
            "excellent": 15.0,
            "good": 10.0,
            "average": 6.0,
            "below_average": 3.0,
            "poor": 1.0
        },
        "weekly_adoption_rate": {
            "excellent": 30.0,
            "good": 20.0,
            "average": 12.0,
            "below_average": 6.0,
            "poor": 3.0
        },
        "monthly_adoption_rate": {
            "excellent": 45.0,
            "good": 35.0,
            "average": 20.0,
            "below_average": 10.0,
            "poor": 5.0
        },
        "yearly_adoption_rate": {
            "excellent": 70.0,
            "good": 55.0,
            "average": 35.0,
            "below_average": 20.0,
            "poor": 10.0
        }
    },
    "crm_systems": {
        "description": "Customer Relationship Management systems",
        "daily_adoption_rate": {
            "excellent": 25.0,
            "good": 15.0,
            "average": 10.0,
            "below_average": 5.0,
            "poor": 2.0
        },
        "weekly_adoption_rate": {
            "excellent": 45.0,
            "good": 30.0,
            "average": 20.0,
            "below_average": 10.0,
            "poor": 5.0
        },
        "monthly_adoption_rate": {
            "excellent": 65.0,
            "good": 50.0,
            "average": 35.0,
            "below_average": 20.0,
            "poor": 10.0
        },
        "yearly_adoption_rate": {
            "excellent": 85.0,
            "good": 70.0,
            "average": 55.0,
            "below_average": 35.0,
            "poor": 20.0
        }
    }
}

# Define recommendation templates based on performance against benchmarks
RECOMMENDATION_TEMPLATES = {
    "excellent": {
        "general": "Your adoption rate is excellent compared to industry benchmarks. Focus on sustaining and building upon your success.",
        "specific_actions": [
            "Document your successful adoption strategies to create organizational knowledge",
            "Look for opportunities to further optimize the user experience",
            "Consider implementing advanced features to maintain user engagement",
            "Develop a formal user champions program to evangelize best practices",
            "Collect and analyze user testimonials to understand what's working well"
        ]
    },
    "good": {
        "general": "Your adoption rate is good compared to industry benchmarks. There's still room for improvement to reach excellence.",
        "specific_actions": [
            "Analyze your most engaged user segments to understand their patterns",
            "Implement targeted campaigns to boost engagement in specific areas",
            "Refine your onboarding process to increase early adoption",
            "Invest in additional training resources to address skill gaps",
            "Collect structured feedback from users to identify enhancement opportunities"
        ]
    },
    "average": {
        "general": "Your adoption rate is average compared to industry benchmarks. Focused improvements could yield significant gains.",
        "specific_actions": [
            "Conduct a comprehensive user experience audit to identify friction points",
            "Implement a formal adoption improvement program with clear KPIs",
            "Revamp your onboarding process with more interactive guidance",
            "Develop targeted training programs for different user segments",
            "Implement regular check-ins with users to gather improvement feedback"
        ]
    },
    "below_average": {
        "general": "Your adoption rate is below average compared to industry benchmarks. This suggests significant opportunities for improvement.",
        "specific_actions": [
            "Conduct user research to understand the primary barriers to adoption",
            "Simplify your user interface to reduce complexity",
            "Implement contextual help and guidance throughout the application",
            "Develop a robust onboarding program with clear user journeys",
            "Create targeted training materials addressing common user challenges"
        ]
    },
    "poor": {
        "general": "Your adoption rate is significantly below industry benchmarks. Urgent attention is needed to improve the user experience and adoption strategy.",
        "specific_actions": [
            "Perform a complete audit of the user experience with external expert help",
            "Redesign your onboarding process with a focus on simplicity and guidance",
            "Implement an adoption recovery program with dedicated resources",
            "Conduct intensive user interviews to understand critical pain points",
            "Consider simplifying or refactoring problematic features based on usage data"
        ]
    }
}

def _determine_benchmark_category(current_rate, benchmark_rates):
    """
    Determine which benchmark category the current rate falls into.
    
    Args:
        current_rate: The current adoption rate
        benchmark_rates: Dictionary of benchmark rates for different categories
    
    Returns:
        str: The benchmark category (excellent, good, average, below_average, poor)
    """
    if current_rate >= benchmark_rates["excellent"]:
        return "excellent"
    elif current_rate >= benchmark_rates["good"]:
        return "good"
    elif current_rate >= benchmark_rates["average"]:
        return "average"
    elif current_rate >= benchmark_rates["below_average"]:
        return "below_average"
    else:
        return "poor"

def _calculate_benchmark_gap(current_rate, benchmark_rates, target_category="good"):
    """
    Calculate the gap between current rate and a target benchmark category.
    
    Args:
        current_rate: The current adoption rate
        benchmark_rates: Dictionary of benchmark rates for different categories
        target_category: The target benchmark category
    
    Returns:
        dict: Dictionary with gap information
    """
    if target_category not in benchmark_rates:
        target_category = "good"  # Default to good if target category not found
    
    target_rate = benchmark_rates[target_category]
    absolute_gap = target_rate - current_rate
    percentage_gap = (absolute_gap / target_rate) * 100 if target_rate > 0 else 0
    
    # Calculate how close to target (as a percentage)
    progress_percentage = (current_rate / target_rate) * 100 if target_rate > 0 else 0
    
    return {
        "target_category": target_category,
        "target_rate": target_rate,
        "absolute_gap": absolute_gap,
        "percentage_gap": percentage_gap,
        "progress_percentage": progress_percentage
    }

def _compare_to_all_benchmarks(current_rate, metric_type):
    """
    Compare the current rate to benchmarks across all industries.
    
    Args:
        current_rate: The current adoption rate
        metric_type: Type of adoption rate (daily, weekly, monthly, yearly)
    
    Returns:
        dict: Dictionary with comparison results for all industries
    """
    rate_field = f"{metric_type}_adoption_rate"
    comparison_results = {}
    
    for industry, data in INDUSTRY_BENCHMARKS.items():
        if rate_field in data:
            benchmark_rates = data[rate_field]
            category = _determine_benchmark_category(current_rate, benchmark_rates)
            gap_to_next = None
            
            # Calculate gap to next level if not already at excellent
            if category != "excellent":
                next_levels = {
                    "poor": "below_average",
                    "below_average": "average",
                    "average": "good",
                    "good": "excellent"
                }
                if category in next_levels:
                    next_level = next_levels[category]
                    gap_to_next = _calculate_benchmark_gap(current_rate, benchmark_rates, next_level)
            
            comparison_results[industry] = {
                "industry_description": data["description"],
                "benchmark_rates": benchmark_rates,
                "current_category": category,
                "gap_to_next_level": gap_to_next
            }
    
    return comparison_results

def _generate_industry_specific_recommendations(comparison_results, preferred_industry=None):
    """
    Generate industry-specific recommendations based on benchmark comparisons.
    
    Args:
        comparison_results: Results from comparing to all benchmarks
        preferred_industry: Preferred industry to focus recommendations on
    
    Returns:
        dict: Dictionary with recommendations for specific industries
    """
    recommendations = {}
    
    # If preferred industry specified, prioritize it
    if preferred_industry and preferred_industry in comparison_results:
        industries_to_process = [preferred_industry] + [i for i in comparison_results if i != preferred_industry]
    else:
        industries_to_process = list(comparison_results.keys())
    
    for industry in industries_to_process:
        result = comparison_results[industry]
        category = result["current_category"]
        
        # Get recommendation templates for this category
        if category in RECOMMENDATION_TEMPLATES:
            template = RECOMMENDATION_TEMPLATES[category]
            
            # Customize general recommendation with industry specifics
            general_rec = template["general"]
            industry_desc = result["industry_description"]
            customized_rec = f"{general_rec} This assessment is based on benchmarks for {industry_desc}."
            
            # Add gap information if available
            gap_info = result.get("gap_to_next_level")
            if gap_info:
                next_level = gap_info["target_category"]
                abs_gap = gap_info["absolute_gap"]
                progress = gap_info["progress_percentage"]
                
                gap_statement = f"To reach {next_level} status, you need to increase by {abs_gap:.2f} "
                gap_statement += f"percentage points. You're currently at {progress:.1f}% of the way there."
                
                customized_rec += f" {gap_statement}"
            
            # Add specific actions
            recommendations[industry] = {
                "general_recommendation": customized_rec,
                "specific_actions": template["specific_actions"],
                "benchmark_category": category
            }
    
    return recommendations

def _find_best_fit_industry(current_rate, metric_type, historical_rates=None):
    """
    Find the industry that best fits the current adoption rate pattern.
    
    Args:
        current_rate: The current adoption rate
        metric_type: Type of adoption rate (daily, weekly, monthly, yearly)
        historical_rates: Optional list of historical rates for pattern matching
    
    Returns:
        str: Name of the best fit industry
    """
    rate_field = f"{metric_type}_adoption_rate"
    best_industry = "saas_overall"  # Default
    closest_distance = float('inf')
    
    # Simple closest match if no historical data
    if not historical_rates:
        for industry, data in INDUSTRY_BENCHMARKS.items():
            if rate_field in data:
                benchmark_rates = data[rate_field]
                # Find closest benchmark category
                for category, rate in benchmark_rates.items():
                    distance = abs(current_rate - rate)
                    if distance < closest_distance:
                        closest_distance = distance
                        best_industry = industry
    
    # More sophisticated pattern matching if historical data available
    else:
        pass  # Could implement time series pattern matching here
    
    return best_industry

def _generate_benchmark_explanation(comparison_results, recommendations, best_fit_industry):
    """
    Generate a comprehensive explanation of the benchmark analysis.
    
    Args:
        comparison_results: Results from comparing to all benchmarks
        recommendations: Recommendations based on benchmark comparisons
        best_fit_industry: The industry that best fits the data
    
    Returns:
        str: Detailed explanation
    """
    if not comparison_results or not recommendations:
        return "Insufficient data to generate benchmark recommendations."
    
    # Focus on the best fit industry
    if best_fit_industry in comparison_results:
        result = comparison_results[best_fit_industry]
        category = result["current_category"]
        industry_desc = result["industry_description"]
        benchmark_rates = result["benchmark_rates"]
    else:
        # Fallback to first industry if best fit not found
        industry = next(iter(comparison_results))
        result = comparison_results[industry]
        category = result["current_category"]
        industry_desc = result["industry_description"]
        benchmark_rates = result["benchmark_rates"]
        best_fit_industry = industry
    
    # Create the explanation
    explanation = f"**Benchmark Analysis: {industry_desc}**\n\n"
    
    # Current standing
    explanation += f"Your current adoption rate is classified as **{category}** "
    explanation += f"compared to benchmarks for {industry_desc}.\n\n"
    
    # Benchmark scale
    explanation += "Industry benchmark scale:\n"
    for level in ["excellent", "good", "average", "below_average", "poor"]:
        if level in benchmark_rates:
            explanation += f"- {level.capitalize()}: {benchmark_rates[level]:.1f}%"
            if level == category:
                explanation += " (your current level)"
            explanation += "\n"
    
    # Gap information
    gap_info = result.get("gap_to_next_level")
    if gap_info and category != "excellent":
        next_level = gap_info["target_category"]
        abs_gap = gap_info["absolute_gap"]
        progress = gap_info["progress_percentage"]
        
        explanation += f"\nTo reach {next_level} status, you need to increase by {abs_gap:.2f} "
        explanation += f"percentage points. You're currently at {progress:.1f}% of the way there.\n"
    
    # Recommendations
    if best_fit_industry in recommendations:
        rec = recommendations[best_fit_industry]
        
        explanation += f"\n**Recommendations:**\n{rec['general_recommendation']}\n\n"
        
        explanation += "Suggested actions:\n"
        for i, action in enumerate(rec["specific_actions"], 1):
            explanation += f"{i}. {action}\n"
    
    # Comparison to other industries
    explanation += "\n**Comparison to Other Industries:**\n"
    for industry, result in comparison_results.items():
        if industry != best_fit_industry:
            explanation += f"- {result['industry_description']}: Your adoption rate is classified as "
            explanation += f"**{result['current_category']}** for this industry\n"
    
    return explanation

def suggest_benchmark_recommendations(data, metric_type="monthly", preferred_industry=None):
    """
    Compare adoption rates to industry benchmarks and suggest recommendations.
    
    Args:
        data: Collection of adoption rate metrics
        metric_type: Type of metric to analyze (daily, weekly, monthly, yearly)
        preferred_industry: Preferred industry for benchmarking
    
    Returns:
        dict: Dictionary containing benchmark comparisons and recommendations
    """
    logger.info(f"Generating benchmark recommendations for {metric_type} adoption rate")
    
    try:
        # Map metric type to field name
        if metric_type == "daily":
            rate_field = "daily_adoption_rate"
        elif metric_type == "weekly":
            rate_field = "weekly_adoption_rate"
        elif metric_type == "yearly":
            rate_field = "yearly_adoption_rate"
        else:  # Default to monthly
            rate_field = "monthly_adoption_rate"
            metric_type = "monthly"
        
        # Sort data by date and get current rate
        sorted_data = sorted(data.overall_adoption_rates, key=lambda x: x.date)
        
        if not sorted_data:
            return {
                "explanation": "No adoption rate data available for benchmark comparison."
            }
        
        current_rate = getattr(sorted_data[-1], rate_field)
        
        # Compare to all industry benchmarks
        comparison_results = _compare_to_all_benchmarks(current_rate, metric_type)
        
        # Determine best fit industry if not specified
        if not preferred_industry or preferred_industry not in INDUSTRY_BENCHMARKS:
            # Get historical rates for pattern matching
            historical_rates = [getattr(item, rate_field) for item in sorted_data]
            best_fit_industry = _find_best_fit_industry(current_rate, metric_type, historical_rates)
        else:
            best_fit_industry = preferred_industry
        
        # Generate recommendations
        recommendations = _generate_industry_specific_recommendations(
            comparison_results, 
            preferred_industry=best_fit_industry
        )
        
        # Generate explanation
        explanation = _generate_benchmark_explanation(
            comparison_results,
            recommendations,
            best_fit_industry
        )
        
        return {
            "current_rate": current_rate,
            "comparison_results": comparison_results,
            "best_fit_industry": best_fit_industry,
            "recommendations": recommendations,
            "explanation": explanation
        }
    
    except Exception as e:
        logger.error(f"Error suggesting benchmark recommendations: {e}")
        return {
            "explanation": f"An error occurred when generating benchmark recommendations: {str(e)}"
        } 