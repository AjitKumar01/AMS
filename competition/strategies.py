"""
Specific implementation of competitive strategies.

Each strategy class implements decision-making logic for:
- Pricing
- Capacity allocation
- Response to competition
"""

from typing import Dict, List, Optional
from datetime import date
import numpy as np

from core.models import Route, FlightDate, Fare
from .airline import Airline, CompetitiveStrategy


class StrategyBase:
    """Base class for competitive strategies."""
    
    def __init__(self, airline: Airline):
        self.airline = airline
    
    def decide_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        competitor_fares: List[float]
    ) -> float:
        """Decide fare for a flight."""
        raise NotImplementedError
    
    def respond_to_competition(
        self,
        route_key: str,
        our_market_share: float,
        competitor_actions: Dict
    ) -> Dict:
        """Respond to competitor actions."""
        raise NotImplementedError


class AggressiveStrategy(StrategyBase):
    """
    Aggressive strategy: Undercut competition to gain market share.
    
    Characteristics:
    - Price 5-10% below lowest competitor
    - Quick to respond to competitor price changes
    - Accept lower yields for higher load factors
    - Willing to operate at thin margins
    """
    
    def decide_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        competitor_fares: List[float]
    ) -> float:
        """Price below competition."""
        base_fare = self.airline.decide_base_fare(route, flight_date)
        
        if competitor_fares:
            # Undercut lowest competitor
            min_competitor = min(competitor_fares)
            target_fare = min_competitor * 0.93  # 7% below
            
            # But don't go below cost
            distance = route.distance_miles
            cost_floor = distance * self.airline.cost_per_seat_mile * 1.05
            
            return max(target_fare, cost_floor, base_fare * 0.85)
        
        return base_fare * 0.95  # Default: 5% below base
    
    def respond_to_competition(
        self,
        route_key: str,
        our_market_share: float,
        competitor_actions: Dict
    ) -> Dict:
        """Respond aggressively to competition."""
        response = {
            'price_adjustment': 1.0,
            'capacity_adjustment': 'maintain',
            'promotion': False
        }
        
        # If losing market share, drop prices
        if our_market_share < 0.3:
            response['price_adjustment'] = 0.90  # Drop 10%
            response['promotion'] = True
        
        # If a competitor dropped prices, match or beat
        for action in competitor_actions.get('price_changes', []):
            if action['change'] < 0:  # Competitor decreased
                response['price_adjustment'] = min(
                    response['price_adjustment'],
                    0.95
                )
        
        return response


class ConservativeStrategy(StrategyBase):
    """
    Conservative strategy: Premium positioning, yield over volume.
    
    Characteristics:
    - Price 10-15% above market average
    - Slow to respond to competition
    - Focus on profitable customers
    - Maintain capacity discipline
    """
    
    def decide_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        competitor_fares: List[float]
    ) -> float:
        """Price above competition for premium positioning."""
        base_fare = self.airline.decide_base_fare(route, flight_date)
        
        if competitor_fares:
            # Price above average
            avg_competitor = np.mean(competitor_fares)
            target_fare = avg_competitor * 1.12  # 12% premium
            
            return max(target_fare, base_fare * 1.10)
        
        return base_fare * 1.15  # Default: 15% above base
    
    def respond_to_competition(
        self,
        route_key: str,
        our_market_share: float,
        competitor_actions: Dict
    ) -> Dict:
        """Maintain position, avoid price wars."""
        response = {
            'price_adjustment': 1.0,
            'capacity_adjustment': 'maintain',
            'promotion': False
        }
        
        # Only respond if market share drops significantly
        if our_market_share < 0.15:
            response['price_adjustment'] = 0.97  # Small adjustment
        
        # Don't follow competitors down, but may adjust up
        max_comp_increase = 0.0
        for action in competitor_actions.get('price_changes', []):
            if action['change'] > 0:
                max_comp_increase = max(max_comp_increase, action['change'])
        
        if max_comp_increase > 0.1:  # If competitors raised >10%
            response['price_adjustment'] = 1.05  # Raise 5%
        
        return response


class MLBasedStrategy(StrategyBase):
    """
    ML-based strategy: Use machine learning for optimization.
    
    Characteristics:
    - Learn from historical data
    - Predict competitor responses
    - Optimize for revenue, not just market share
    - Adaptive pricing based on demand patterns
    """
    
    def __init__(self, airline: Airline):
        super().__init__(airline)
        self.learning_rate = 0.1
        self.price_elasticity = -1.5  # Learned parameter
    
    def decide_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        competitor_fares: List[float]
    ) -> float:
        """Use ML model for pricing."""
        base_fare = self.airline.decide_base_fare(route, flight_date)
        
        # Features for ML model (simplified - in reality use trained model)
        days_to_departure = (flight_date.departure_date - date.today()).days
        current_load_factor = flight_date.load_factor()
        
        # Demand-based adjustment
        if days_to_departure > 0:
            # High load factor far out = increase price
            if current_load_factor > 0.7 and days_to_departure > 21:
                multiplier = 1.15
            # Low load factor close in = decrease price
            elif current_load_factor < 0.4 and days_to_departure < 14:
                multiplier = 0.90
            else:
                multiplier = 1.0
        else:
            multiplier = 1.0
        
        # Competitive adjustment
        if competitor_fares:
            avg_comp = np.mean(competitor_fares)
            # Price slightly below average if we have capacity
            if current_load_factor < 0.6:
                target = avg_comp * 0.97
            else:
                target = avg_comp * 1.03
            
            return target * multiplier
        
        return base_fare * multiplier
    
    def respond_to_competition(
        self,
        route_key: str,
        our_market_share: float,
        competitor_actions: Dict
    ) -> Dict:
        """Use reinforcement learning approach."""
        response = {
            'price_adjustment': 1.0,
            'capacity_adjustment': 'maintain',
            'promotion': False
        }
        
        # Explore-exploit tradeoff
        if np.random.random() < 0.1:  # 10% exploration
            response['price_adjustment'] = np.random.uniform(0.92, 1.08)
        else:  # Exploit learned policy
            # Adjust based on market share gradient
            if our_market_share < 0.25:
                response['price_adjustment'] = 0.95
            elif our_market_share > 0.40:
                response['price_adjustment'] = 1.05
        
        return response


class MatchCompetitorStrategy(StrategyBase):
    """
    Match competitor strategy: Follow the market leader.
    
    Characteristics:
    - Match prices of market leader
    - Copy capacity adjustments
    - Minimize risk
    - Focus on operational efficiency
    """
    
    def decide_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        competitor_fares: List[float]
    ) -> float:
        """Match competitor prices."""
        base_fare = self.airline.decide_base_fare(route, flight_date)
        
        if competitor_fares:
            # Match the median price
            median_price = np.median(competitor_fares)
            return median_price
        
        return base_fare
    
    def respond_to_competition(
        self,
        route_key: str,
        our_market_share: float,
        competitor_actions: Dict
    ) -> Dict:
        """Mirror competitor actions."""
        response = {
            'price_adjustment': 1.0,
            'capacity_adjustment': 'maintain',
            'promotion': False
        }
        
        # Match average price change
        price_changes = [
            a['change'] for a in competitor_actions.get('price_changes', [])
        ]
        if price_changes:
            avg_change = np.mean(price_changes)
            response['price_adjustment'] = 1.0 + avg_change
        
        return response


class YieldFocusedStrategy(StrategyBase):
    """
    Yield-focused strategy: Maximize revenue per passenger.
    
    Characteristics:
    - High prices, accept lower load factors
    - Target business travelers
    - Restrict low-fare inventory
    - Focus on profitable segments
    """
    
    def decide_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        competitor_fares: List[float]
    ) -> float:
        """Set high fares for yield optimization."""
        base_fare = self.airline.decide_base_fare(route, flight_date)
        
        # Always price at premium
        premium_fare = base_fare * 1.25
        
        if competitor_fares:
            max_comp = max(competitor_fares)
            # Price at or above highest competitor
            return max(premium_fare, max_comp * 1.02)
        
        return premium_fare
    
    def respond_to_competition(
        self,
        route_key: str,
        our_market_share: float,
        competitor_actions: Dict
    ) -> Dict:
        """Maintain premium positioning."""
        response = {
            'price_adjustment': 1.0,
            'capacity_adjustment': 'maintain',
            'promotion': False
        }
        
        # Only respond if losing significant high-yield customers
        # (would need customer segment tracking)
        
        return response


class MarketShareStrategy(StrategyBase):
    """
    Market share strategy: Maximize passenger volume.
    
    Characteristics:
    - Aggressive pricing
    - High capacity utilization
    - Promotional activities
    - Build customer base for long-term
    """
    
    def decide_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        competitor_fares: List[float]
    ) -> float:
        """Price for maximum volume."""
        base_fare = self.airline.decide_base_fare(route, flight_date)
        
        if competitor_fares:
            # Beat all competitors
            min_comp = min(competitor_fares)
            return min_comp * 0.95
        
        return base_fare * 0.90
    
    def respond_to_competition(
        self,
        route_key: str,
        our_market_share: float,
        competitor_actions: Dict
    ) -> Dict:
        """Aggressively pursue market share."""
        response = {
            'price_adjustment': 0.95,  # Always push prices down
            'capacity_adjustment': 'increase',
            'promotion': True
        }
        
        # If gaining share, maintain momentum
        if our_market_share > 0.35:
            response['capacity_adjustment'] = 'increase'
        
        return response
