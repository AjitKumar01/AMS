"""
Airline agent implementation for competitive simulation.

Each airline is an autonomous agent that:
- Manages its own fleet and schedules
- Makes pricing and capacity decisions
- Responds to competitor actions
- Tracks performance metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from datetime import date, datetime
from enum import Enum
import numpy as np

from core.models import (
    FlightSchedule, Airport, Route, Aircraft, 
    FlightDate, Fare, RMControl
)


class CompetitiveStrategy(Enum):
    """Competitive strategies for airline agents."""
    AGGRESSIVE = "aggressive"  # Price low, fill planes
    CONSERVATIVE = "conservative"  # Price high, premium positioning
    ML_BASED = "ml_based"  # Use ML to optimize decisions
    MATCH_COMPETITOR = "match_competitor"  # Match competitor prices
    YIELD_FOCUSED = "yield_focused"  # Maximize yield over load factor
    MARKET_SHARE = "market_share"  # Maximize market share


@dataclass
class AirlinePerformance:
    """Track airline performance metrics."""
    airline_code: str
    total_revenue: float = 0.0
    total_bookings: int = 0
    total_passengers: int = 0
    total_capacity: int = 0
    cancellations: int = 0
    denied_boardings: int = 0
    flights_operated: int = 0
    
    # Market metrics
    market_share_bookings: float = 0.0  # % of bookings in market
    market_share_revenue: float = 0.0   # % of revenue in market
    
    # Financial metrics
    average_fare: float = 0.0
    yield_per_passenger_mile: float = 0.0
    revenue_per_available_seat_mile: float = 0.0
    
    # Operational metrics
    load_factor: float = 0.0
    spill_rate: float = 0.0  # % of demand turned away
    
    # Competitive metrics
    price_index: float = 1.0  # Relative to market average
    quality_index: float = 1.0  # Schedule quality
    
    def update_metrics(self):
        """Calculate derived metrics."""
        if self.total_bookings > 0:
            self.average_fare = self.total_revenue / self.total_passengers
        
        if self.total_capacity > 0:
            self.load_factor = self.total_passengers / self.total_capacity
            self.revenue_per_available_seat_mile = self.total_revenue / self.total_capacity


@dataclass
class CompetitorIntelligence:
    """Information about competitor airline."""
    airline_code: str
    observed_fares: Dict[str, List[float]] = field(default_factory=dict)  # route -> fares
    observed_load_factors: Dict[str, List[float]] = field(default_factory=dict)
    observed_capacity: Dict[str, int] = field(default_factory=dict)
    last_price_change: Dict[str, datetime] = field(default_factory=dict)
    estimated_market_share: float = 0.0
    
    def get_average_fare(self, route_key: str) -> float:
        """Get average fare for a route."""
        if route_key in self.observed_fares and self.observed_fares[route_key]:
            return np.mean(self.observed_fares[route_key])
        return 0.0
    
    def get_recent_fare(self, route_key: str, n: int = 5) -> float:
        """Get recent average fare for a route."""
        if route_key in self.observed_fares and self.observed_fares[route_key]:
            recent = self.observed_fares[route_key][-n:]
            return np.mean(recent)
        return 0.0


class Airline:
    """
    Airline agent with competitive behavior.
    
    Each airline:
    - Operates flights according to its schedule
    - Manages inventory and pricing independently
    - Observes competitor actions and responds
    - Uses different strategies to compete
    """
    
    def __init__(
        self,
        code: str,
        name: str,
        strategy: CompetitiveStrategy,
        base_price_multiplier: float = 1.0,
        quality_multiplier: float = 1.0,
        cost_per_seat_mile: float = 0.10,
        brand_preference: float = 0.0
    ):
        """
        Initialize airline agent.
        
        Args:
            code: Airline code (e.g., "AA", "UA", "DL")
            name: Full airline name
            strategy: Competitive strategy to use
            base_price_multiplier: Base price level (1.0 = market average)
            quality_multiplier: Service quality perception
            cost_per_seat_mile: Operating cost per seat mile
            brand_preference: Customer brand preference (-1 to 1)
        """
        self.code = code
        self.name = name
        self.strategy = strategy
        self.base_price_multiplier = base_price_multiplier
        self.quality_multiplier = quality_multiplier
        self.cost_per_seat_mile = cost_per_seat_mile
        self.brand_preference = brand_preference
        
        # Operational data
        self.schedules: List[FlightSchedule] = []
        self.aircraft: List[Aircraft] = []
        self.flight_dates: Dict[str, FlightDate] = {}
        
        # Performance tracking
        self.performance = AirlinePerformance(airline_code=code)
        
        # Competitive intelligence
        self.competitors: Dict[str, CompetitorIntelligence] = {}
        
        # Strategy parameters (learned or set)
        self.price_sensitivity = 0.15  # How much to respond to competitor prices
        self.capacity_response_threshold = 0.85  # Load factor to trigger capacity change
        self.price_update_frequency_days = 7  # How often to review prices
        
        # Historical tracking for learning
        self.price_history: Dict[str, List[tuple]] = {}  # route -> [(date, price, outcome)]
        self.demand_history: Dict[str, List[tuple]] = {}  # route -> [(date, demand, bookings)]
    
    def add_schedule(self, schedule: FlightSchedule):
        """Add a flight schedule to this airline."""
        self.schedules.append(schedule)
    
    def add_aircraft(self, aircraft: Aircraft):
        """Add aircraft to fleet."""
        if aircraft not in self.aircraft:
            self.aircraft.append(aircraft)
    
    def observe_competitor_fare(
        self, 
        competitor_code: str, 
        route_key: str, 
        fare: float,
        timestamp: datetime
    ):
        """Observe and record competitor fare."""
        if competitor_code not in self.competitors:
            self.competitors[competitor_code] = CompetitorIntelligence(competitor_code)
        
        competitor = self.competitors[competitor_code]
        if route_key not in competitor.observed_fares:
            competitor.observed_fares[route_key] = []
        
        competitor.observed_fares[route_key].append(fare)
        competitor.last_price_change[route_key] = timestamp
        
        # Keep only recent history (last 30 observations)
        if len(competitor.observed_fares[route_key]) > 30:
            competitor.observed_fares[route_key] = competitor.observed_fares[route_key][-30:]
    
    def observe_competitor_load_factor(
        self,
        competitor_code: str,
        route_key: str,
        load_factor: float
    ):
        """Observe competitor load factor."""
        if competitor_code not in self.competitors:
            self.competitors[competitor_code] = CompetitorIntelligence(competitor_code)
        
        competitor = self.competitors[competitor_code]
        if route_key not in competitor.observed_load_factors:
            competitor.observed_load_factors[route_key] = []
        
        competitor.observed_load_factors[route_key].append(load_factor)
        
        # Keep only recent history
        if len(competitor.observed_load_factors[route_key]) > 30:
            competitor.observed_load_factors[route_key] = (
                competitor.observed_load_factors[route_key][-30:]
            )
    
    def decide_base_fare(
        self,
        route: Route,
        flight_date: FlightDate,
        current_fare: Optional[Fare] = None
    ) -> float:
        """
        Decide base fare based on strategy and competition.
        
        Returns:
            Base fare amount
        """
        route_key = f"{route.origin.code}-{route.destination.code}"
        
        # Calculate cost-based floor
        distance = route.distance_miles
        seats = flight_date.schedule.aircraft.total_capacity
        cost_floor = distance * seats * self.cost_per_seat_mile / seats
        
        # Base market price (simple distance-based)
        market_base = 50 + distance * 0.15
        
        # Apply airline's base multiplier
        base_price = market_base * self.base_price_multiplier
        
        # Get competitor prices
        competitor_prices = []
        for comp_code, intel in self.competitors.items():
            avg_fare = intel.get_recent_fare(route_key, n=5)
            if avg_fare > 0:
                competitor_prices.append(avg_fare)
        
        # Apply strategy-specific adjustments
        if self.strategy == CompetitiveStrategy.AGGRESSIVE:
            # Price below competition
            if competitor_prices:
                min_comp = min(competitor_prices)
                base_price = min(base_price, min_comp * 0.95)
        
        elif self.strategy == CompetitiveStrategy.CONSERVATIVE:
            # Price above competition, premium positioning
            if competitor_prices:
                max_comp = max(competitor_prices)
                base_price = max(base_price, max_comp * 1.05)
        
        elif self.strategy == CompetitiveStrategy.MATCH_COMPETITOR:
            # Match average competitor price
            if competitor_prices:
                base_price = np.mean(competitor_prices)
        
        elif self.strategy == CompetitiveStrategy.YIELD_FOCUSED:
            # Higher prices, accept lower load factor
            base_price *= 1.2
        
        elif self.strategy == CompetitiveStrategy.MARKET_SHARE:
            # Lower prices to capture share
            base_price *= 0.9
            if competitor_prices:
                base_price = min(base_price, min(competitor_prices) * 0.98)
        
        # Ensure price covers costs
        base_price = max(base_price, cost_floor * 1.1)
        
        # Dynamic adjustments based on current performance
        if current_fare:
            # Check current load factor
            load_factor = flight_date.load_factor()
            days_to_departure = (flight_date.departure_date - date.today()).days
            
            if days_to_departure > 0:
                # Adjust based on booking pace
                if load_factor > 0.8 and days_to_departure > 14:
                    # Selling fast, increase price
                    base_price *= 1.1
                elif load_factor < 0.3 and days_to_departure < 21:
                    # Selling slow, decrease price
                    base_price *= 0.95
        
        return round(base_price, 2)
    
    def should_update_prices(self, current_date: date, route_key: str) -> bool:
        """Determine if prices should be updated."""
        if route_key not in self.price_history:
            return True
        
        # Check if enough time has passed
        last_updates = [h[0] for h in self.price_history[route_key]]
        if last_updates:
            days_since_update = (current_date - max(last_updates)).days
            return days_since_update >= self.price_update_frequency_days
        
        return True
    
    def evaluate_capacity_needs(self, route_key: str) -> str:
        """
        Evaluate if capacity should be increased/decreased.
        
        Returns:
            "increase", "decrease", or "maintain"
        """
        # Look at recent load factors
        recent_flights = []
        for fd in self.flight_dates.values():
            fd_route_key = f"{fd.schedule.route.origin.code}-{fd.schedule.route.destination.code}"
            if fd_route_key == route_key:
                recent_flights.append(fd)
        
        if not recent_flights:
            return "maintain"
        
        # Calculate average load factor
        avg_lf = np.mean([fd.load_factor() for fd in recent_flights[-10:]])
        
        # Decision based on strategy
        if self.strategy in [CompetitiveStrategy.AGGRESSIVE, CompetitiveStrategy.MARKET_SHARE]:
            # More willing to add capacity
            if avg_lf > 0.85:
                return "increase"
            elif avg_lf < 0.50:
                return "decrease"
        
        elif self.strategy in [CompetitiveStrategy.CONSERVATIVE, CompetitiveStrategy.YIELD_FOCUSED]:
            # More conservative with capacity
            if avg_lf > 0.90:
                return "increase"
            elif avg_lf < 0.40:
                return "decrease"
        
        return "maintain"
    
    def record_booking(
        self,
        route_key: str,
        booking_date: date,
        revenue: float,
        passengers: int
    ):
        """Record a successful booking for learning."""
        if route_key not in self.demand_history:
            self.demand_history[route_key] = []
        
        self.demand_history[route_key].append((booking_date, passengers, revenue))
        
        # Update performance
        self.performance.total_revenue += revenue
        self.performance.total_bookings += 1
        self.performance.total_passengers += passengers
    
    def record_spill(self, route_key: str, passengers: int):
        """Record demand that couldn't be accommodated."""
        # Track spilled demand for capacity planning
        pass
    
    def get_market_position(self, route_key: str) -> Dict[str, float]:
        """
        Analyze market position on a route.
        
        Returns:
            Dictionary with position metrics
        """
        position = {
            'our_avg_fare': 0.0,
            'competitor_avg_fare': 0.0,
            'our_load_factor': 0.0,
            'competitor_load_factor': 0.0,
            'price_index': 1.0,
            'capacity_share': 0.0
        }
        
        # Calculate our metrics
        our_flights = [
            fd for fd in self.flight_dates.values()
            if f"{fd.schedule.route.origin.code}-{fd.schedule.route.destination.code}" == route_key
        ]
        
        if our_flights:
            position['our_load_factor'] = np.mean([fd.load_factor() for fd in our_flights])
        
        # Competitor metrics
        competitor_fares = []
        for intel in self.competitors.values():
            fare = intel.get_recent_fare(route_key)
            if fare > 0:
                competitor_fares.append(fare)
        
        if competitor_fares:
            position['competitor_avg_fare'] = np.mean(competitor_fares)
            if position['our_avg_fare'] > 0:
                position['price_index'] = (
                    position['our_avg_fare'] / position['competitor_avg_fare']
                )
        
        return position
    
    def __repr__(self) -> str:
        return (
            f"Airline(code={self.code}, name={self.name}, "
            f"strategy={self.strategy.value}, "
            f"flights={len(self.schedules)})"
        )
