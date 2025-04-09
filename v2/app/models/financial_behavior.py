"""
Financial Behavior Module
Analyzes user financial behavior patterns to ensure responsible recommendations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class FinancialBehaviorAnalyzer:
    """
    Analyzes user financial behavior and provides responsible recommendation guidelines
    """
    
    def __init__(self, recommendation_engine):
        """
        Initialize the financial behavior analyzer with the recommendation engine
        """
        self.engine = recommendation_engine
        
        # Define spending thresholds and categories
        self.spending_thresholds = {
            'low': 100,
            'medium': 500,
            'high': 1000
        }
        
        # Define luxury categories
        self.luxury_categories = [
            'Luxury Watches',
            'Designer Clothing',
            'High-End Electronics',
            'Jewelry',
            'Premium Accessories'
        ]
        
        # Placeholder for user purchase history (in a real system, this would come from a database)
        # This would track user purchases including items, prices, dates, etc.
        self.user_purchase_history = {}
    
    def add_purchase(self, user_id, item_id, price, timestamp=None):
        """
        Record a user purchase to analyze spending patterns
        """
        user_id = int(user_id)
        item_id = int(item_id)
        price = float(price)
        timestamp = timestamp or datetime.now()
        
        if user_id not in self.user_purchase_history:
            self.user_purchase_history[user_id] = []
        
        # Get item details
        item_details = self.engine.get_item_details(item_id)
        
        # Record the purchase
        self.user_purchase_history[user_id].append({
            'item_id': item_id,
            'price': price,
            'timestamp': timestamp,
            'category': item_details.get('main_category', 'Unknown'),
            'subcategory': item_details.get('subcategory', 'Unknown')
        })
    
    def get_user_spending_profile(self, user_id):
        """
        Analyze user spending profile based on purchase history
        """
        user_id = int(user_id)
        
        try:
            # Get user details for income-based calculations
            user_details = None
            for user in self.engine.get_all_users():
                if user['user_id'] == user_id:
                    user_details = user
                    break
            
            # Use income bracket to determine default spending profile
            income_bracket = user_details.get('income_bracket', 'middle') if user_details else 'middle'
            
            # Map income brackets to risk tolerance and spending levels
            income_to_risk = {
                'under_25k': 'conservative',
                '25k-50k': 'cautious',
                '50k-75k': 'moderate',
                '75k-100k': 'balanced',
                '100k-150k': 'growth-oriented', 
                'over_150k': 'aggressive'
            }
            
            # Default profile for users with no purchase history - based on income bracket
            default_profile = {
                'user_id': user_id,
                'monthly_spending': 0.0,
                'spending_level': self._get_default_spending_level(income_bracket),
                'luxury_ratio': 0.0,
                'spending_frequency': 'new_user',
                'average_price': 0.0,
                'price_ceiling': self._get_default_price_ceiling(income_bracket),
                'risk_tolerance': income_to_risk.get(income_bracket, 'moderate'),
                'budget_status': 'balanced'
            }
            
            # If no purchase history, return default profile with income-based values
            if user_id not in self.user_purchase_history or not self.user_purchase_history[user_id]:
                print(f"No purchase history for user {user_id}, returning default profile")
                return default_profile
            
            # Get user's purchase history
            purchases = self.user_purchase_history[user_id]
            
            # Calculate total spent
            total_spent = sum(purchase['price'] for purchase in purchases)
            
            # Calculate spending in the last 30 days
            now = datetime.now()
            last_month = now - timedelta(days=30)
            monthly_purchases = [p for p in purchases if p['timestamp'] >= last_month]
            monthly_spending = sum(p['price'] for p in monthly_purchases)
            
            # Determine spending level
            if monthly_spending <= self.spending_thresholds['low']:
                spending_level = 'low'
                price_ceiling = self.spending_thresholds['low']
            elif monthly_spending <= self.spending_thresholds['medium']:
                spending_level = 'medium'
                price_ceiling = self.spending_thresholds['medium']
            else:
                spending_level = 'high'
                price_ceiling = self.spending_thresholds['high'] * 1.5  # Allow higher ceiling for high spenders
            
            # Calculate luxury purchase ratio
            luxury_purchases = [p for p in purchases if p['subcategory'] in self.luxury_categories]
            luxury_ratio = len(luxury_purchases) / len(purchases) if purchases else 0
            
            # Determine purchase frequency
            purchase_dates = [p['timestamp'] for p in purchases]
            if len(purchase_dates) < 2:
                spending_frequency = 'first_time'
            else:
                # Sort purchases by date
                purchase_dates.sort()
                time_diffs = [(purchase_dates[i] - purchase_dates[i-1]).days for i in range(1, len(purchase_dates))]
                avg_time_between_purchases = sum(time_diffs) / len(time_diffs)
                
                if avg_time_between_purchases < 7:
                    spending_frequency = 'frequent'
                elif avg_time_between_purchases < 30:
                    spending_frequency = 'regular'
                else:
                    spending_frequency = 'occasional'
            
            # Calculate average price of purchases
            average_price = total_spent / len(purchases)
            
            # Determine budget status based on spending vs expected spending for income bracket
            budget_status = self._calculate_budget_status(monthly_spending, income_bracket)
            
            # Determine risk tolerance based on spending patterns and income
            risk_tolerance = self._calculate_risk_tolerance(luxury_ratio, spending_level, income_bracket)
            
            # Create spending profile
            spending_profile = {
                'user_id': user_id,
                'monthly_spending': monthly_spending,
                'spending_level': spending_level,
                'luxury_ratio': luxury_ratio,
                'spending_frequency': spending_frequency,
                'average_price': average_price,
                'price_ceiling': price_ceiling,
                'risk_tolerance': risk_tolerance,
                'budget_status': budget_status
            }
            
            return spending_profile
        
        except Exception as e:
            print(f"Error getting spending profile: {str(e)}")
            # Return a safe default profile if anything goes wrong
            return {
                'user_id': user_id,
                'spending_level': 'moderate',
                'risk_tolerance': 'balanced',
                'budget_status': 'balanced'
            }
    
    def _get_default_spending_level(self, income_bracket):
        """Helper method to determine default spending level based on income"""
        income_to_spending_level = {
            'under_25k': 'low',
            '25k-50k': 'low',
            '50k-75k': 'medium',
            '75k-100k': 'medium',
            '100k-150k': 'high',
            'over_150k': 'high'
        }
        return income_to_spending_level.get(income_bracket, 'medium')
    
    def _get_default_price_ceiling(self, income_bracket):
        """Helper method to determine default price ceiling based on income"""
        income_to_ceiling = {
            'under_25k': self.spending_thresholds['low'] * 0.5,
            '25k-50k': self.spending_thresholds['low'],
            '50k-75k': self.spending_thresholds['medium'] * 0.7,
            '75k-100k': self.spending_thresholds['medium'],
            '100k-150k': self.spending_thresholds['high'] * 0.8,
            'over_150k': self.spending_thresholds['high']
        }
        return income_to_ceiling.get(income_bracket, self.spending_thresholds['medium'])
    
    def _calculate_budget_status(self, monthly_spending, income_bracket):
        """Calculate budget status based on spending vs expected spending for income bracket"""
        # Expected monthly spending by income bracket (estimated)
        expected_spending = {
            'under_25k': 1000,
            '25k-50k': 1500,
            '50k-75k': 2000,
            '75k-100k': 3000,
            '100k-150k': 4000,
            'over_150k': 5000
        }
        
        expected = expected_spending.get(income_bracket, 2500)
        
        # No spending yet
        if monthly_spending == 0:
            return 'balanced'
        
        # Calculate the ratio of actual to expected spending
        ratio = monthly_spending / expected
        
        if ratio < 0.5:
            return 'under budget'
        elif ratio < 0.8:
            return 'within budget'
        elif ratio < 1.2:
            return 'balanced'
        elif ratio < 1.5:
            return 'approaching limit'
        else:
            return 'over budget'
    
    def _calculate_risk_tolerance(self, luxury_ratio, spending_level, income_bracket):
        """Calculate risk tolerance based on spending patterns and income"""
        # Base risk tolerance on income bracket
        base_tolerance = {
            'under_25k': 'conservative',
            '25k-50k': 'cautious',
            '50k-75k': 'moderate',
            '75k-100k': 'balanced',
            '100k-150k': 'growth-oriented',
            'over_150k': 'aggressive'
        }.get(income_bracket, 'moderate')
        
        # Adjust based on luxury ratio and spending level
        if luxury_ratio > 0.5 and spending_level == 'high':
            # Likes luxury items and spends a lot
            return 'aggressive'
        elif luxury_ratio < 0.1 and spending_level == 'low':
            # Avoids luxury items and spends little
            return 'conservative'
        
        return base_tolerance
    
    def filter_recommendations(self, user_id, recommendations):
        """
        Filter and rerank recommendations based on responsible financial guidelines
        """
        user_id = int(user_id)
        
        # Get user's spending profile
        profile = self.get_user_spending_profile(user_id)
        
        # Filter out items that are too expensive
        price_ceiling = profile['price_ceiling']
        filtered_recommendations = []
        
        for item in recommendations:
            item_price = item.get('price', 0)
            
            # Always include items below user's average purchase price
            if item_price <= profile['average_price']:
                filtered_recommendations.append(item)
                continue
            
            # For items above average price but below ceiling, calculate an "affordability score"
            if item_price <= price_ceiling:
                # How close is this to the ceiling (0 = at average price, 1 = at ceiling)
                price_position = (item_price - profile['average_price']) / (price_ceiling - profile['average_price']) if price_ceiling > profile['average_price'] else 0
                
                # Calculate affordability penalty (higher = more penalty)
                affordability_penalty = price_position * 0.5
                
                # Adjust score based on affordability
                adjusted_score = item['score'] * (1 - affordability_penalty)
                
                # Add affordability information
                affordability_info = {
                    'original_score': item['score'],
                    'affordability_adjusted_score': adjusted_score,
                    'affordability_factor': 1 - affordability_penalty,
                    'price_position': price_position
                }
                
                # Update item score
                item['score'] = adjusted_score
                item['affordability_info'] = affordability_info
                
                filtered_recommendations.append(item)
            else:
                # For luxury items above price ceiling, only include if user has luxury purchase history
                if item_price > price_ceiling:
                    # Check if it's a luxury item
                    is_luxury = item.get('subcategory', '') in self.luxury_categories
                    
                    # Only include luxury items for users with luxury purchase history
                    if is_luxury and profile['luxury_ratio'] > 0.2:
                        # Add a warning
                        item['luxury_warning'] = True
                        # Significantly reduce score for very expensive items
                        item['score'] *= 0.5
                        filtered_recommendations.append(item)
        
        # Re-sort based on adjusted scores
        filtered_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return filtered_recommendations
    
    def analyze_spending_patterns(self, user_id):
        """
        Analyze spending patterns to detect potential issues
        """
        user_id = int(user_id)
        
        # Default analysis for users with no purchase history
        default_analysis = {
            'warnings': [],
            'responsible_spending': True
        }
        
        # If no purchase history, return default analysis
        if user_id not in self.user_purchase_history or not self.user_purchase_history[user_id]:
            return default_analysis
        
        # Get user's purchase history
        purchases = self.user_purchase_history[user_id]
        
        # Calculate metrics
        now = datetime.now()
        monthly_purchases = [p for p in purchases if p['timestamp'] >= (now - timedelta(days=30))]
        weekly_purchases = [p for p in purchases if p['timestamp'] >= (now - timedelta(days=7))]
        
        monthly_spend = sum(p['price'] for p in monthly_purchases)
        weekly_spend = sum(p['price'] for p in weekly_purchases)
        
        # Initialize analysis
        analysis = {
            'warnings': [],
            'responsible_spending': True
        }
        
        # Check for rapid increase in spending
        if weekly_spend > monthly_spend * 0.5 and len(weekly_purchases) > 3:
            analysis['warnings'].append({
                'type': 'rapid_spending_increase',
                'message': 'Significant increase in spending over the past week.',
                'severity': 'medium'
            })
        
        # Check for frequent large purchases
        large_purchases = [p for p in weekly_purchases if p['price'] > self.spending_thresholds['medium']]
        if len(large_purchases) > 2:
            analysis['warnings'].append({
                'type': 'frequent_large_purchases',
                'message': 'Multiple large purchases in a short time period.',
                'severity': 'high'
            })
            analysis['responsible_spending'] = False
        
        # Check for excessive spending
        if monthly_spend > self.spending_thresholds['high'] * 2:
            analysis['warnings'].append({
                'type': 'excessive_spending',
                'message': 'Monthly spending exceeds recommended limits.',
                'severity': 'high'
            })
            analysis['responsible_spending'] = False
        
        return analysis
    
    def get_responsible_spending_advice(self, user_id):
        """
        Provide responsible spending advice based on user's financial behavior
        """
        user_id = int(user_id)
        
        try:
            # Analyze spending patterns
            analysis = self.analyze_spending_patterns(user_id)
            
            # Get user's profile
            profile = self.get_user_spending_profile(user_id)
            
            # Default advice (even for users with no warnings)
            advice = {
                'user_id': user_id,
                'warnings': analysis.get('warnings', []),
                'responsible': analysis.get('responsible_spending', True),
                'recommendations': [],
                'profile_summary': {
                    'spending_level': profile.get('spending_level', 'moderate'),
                    'risk_tolerance': profile.get('risk_tolerance', 'balanced'),
                    'budget_status': profile.get('budget_status', 'balanced')
                }
            }
            
            # For new users with no purchase history, provide generic advice
            if user_id not in self.user_purchase_history or not self.user_purchase_history[user_id]:
                advice['recommendations'] = [
                    "Welcome! We'll provide personalized financial insights as you make purchases.",
                    "Consider setting a budget for different product categories based on your income.",
                    "Start with smaller purchases to build your spending profile."
                ]
                
                # Add a gentle warning for high-income users about luxury items
                user_details = next((u for u in self.engine.get_all_users() if u['user_id'] == user_id), None)
                if user_details and user_details.get('income_bracket') in ['100k-150k', 'over_150k']:
                    advice['warnings'].append({
                        'type': 'high_income_guidance',
                        'message': 'With your income bracket, you have access to luxury items, but remember to maintain a balanced approach to spending.',
                        'severity': 'low'
                    })
                
                return advice
            
            # Add specific recommendations based on warning types
            for warning in analysis['warnings']:
                if warning['type'] == 'rapid_spending_increase':
                    advice['recommendations'].append(
                        'Consider reviewing your recent purchases and creating a budget for the rest of the month.'
                    )
                elif warning['type'] == 'frequent_large_purchases':
                    advice['recommendations'].append(
                        'It may be helpful to space out large purchases over time to maintain financial balance.'
                    )
                elif warning['type'] == 'excessive_spending':
                    advice['recommendations'].append(
                        'Your spending this month is higher than usual. Consider setting spending limits for different categories.'
                    )
            
            # If no specific warnings, provide general advice based on profile
            if not advice['recommendations']:
                if profile['spending_level'] == 'high':
                    advice['recommendations'].append(
                        'Your spending level is relatively high. Consider reviewing your purchase patterns regularly.'
                    )
                elif profile['spending_level'] == 'low' and profile['spending_frequency'] != 'first_time':
                    advice['recommendations'].append(
                        'You maintain a conservative spending approach, which is great for long-term financial health.'
                    )
                else:
                    advice['recommendations'].append(
                        'Your spending patterns appear balanced. Continue monitoring your purchases to maintain financial health.'
                    )
            
            return advice
            
        except Exception as e:
            print(f"Error getting spending advice: {str(e)}")
            # Return safe default advice if anything goes wrong
            return {
                'user_id': user_id,
                'warnings': [],
                'responsible': True,
                'recommendations': ["We'll provide personalized financial insights as you make purchases."],
                'profile_summary': {
                    'spending_level': 'moderate',
                    'risk_tolerance': 'balanced',
                    'budget_status': 'balanced'
                }
            } 