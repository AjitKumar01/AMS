"""
Overbooking optimization for airline revenue management.

Implements:
- No-show probability modeling by segment, route, booking class
- Overbooking limit calculation using critical fractile policy
- Denied boarding cost estimation
- Show-up distribution forecasting
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date, datetime
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar
from enum import Enum

from core.models import (
    CustomerSegment, BookingClass, FlightDate, 
    Route, Booking
)


class OverbookingMethod(Enum):
    """Methods for calculating overbooking limits."""
    CRITICAL_FRACTILE = "critical_fractile"  # Standard industry approach
    EMSR_OB = "emsr_ob"  # EMSR adapted for overbooking
    RISK_AVERSE = "risk_averse"  # Conservative, minimize denied boarding
    REVENUE_MAXIMIZING = "revenue_maximizing"  # Aggressive, maximize revenue


@dataclass
class NoShowModel:
    """
    Model for no-show probability estimation.
    
    No-show rates vary by:
    - Customer segment (business has higher no-show than leisure)
    - Booking class (higher fare classes have lower no-show)
    - Advance purchase (last-minute bookings have lower no-show)
    - Route characteristics (hub vs point-to-point)
    - Day of week and time of day
    """
    
    # Base no-show rates by segment
    base_rates: Dict[CustomerSegment, float] = field(default_factory=lambda: {
        CustomerSegment.BUSINESS: 0.15,  # 15% no-show for business
        CustomerSegment.LEISURE: 0.08,   # 8% no-show for leisure
        CustomerSegment.VFR: 0.06,       # 6% no-show for VFR
        CustomerSegment.GROUP: 0.03      # 3% no-show for groups
    })
    
    # Fare class adjustments (multipliers)
    fare_class_multipliers: Dict[BookingClass, float] = field(default_factory=lambda: {
        BookingClass.Y: 1.5,   # Full-fare economy: higher no-show
        BookingClass.B: 1.3,
        BookingClass.M: 1.0,   # Mid-range: baseline
        BookingClass.H: 0.8,
        BookingClass.Q: 0.7,
        BookingClass.K: 0.6,
        BookingClass.L: 0.5,   # Lowest fare: lowest no-show (non-refundable)
        BookingClass.J: 1.4,   # Business class
        BookingClass.C: 1.3,
        BookingClass.F: 1.6    # First class: highest no-show
    })
    
    # Historical data for learning
    historical_no_shows: List[Tuple[date, str, int, int]] = field(default_factory=list)
    # Format: (flight_date, route_key, bookings, shows)
    
    def get_no_show_probability(
        self,
        segment: CustomerSegment,
        booking_class: BookingClass,
        advance_purchase_days: int,
        route_distance: float,
        is_hub_flight: bool = False
    ) -> float:
        """
        Calculate no-show probability for a booking.
        
        Args:
            segment: Customer segment
            booking_class: Fare class
            advance_purchase_days: Days between booking and departure
            route_distance: Flight distance in miles
            is_hub_flight: Whether flight operates through a hub
            
        Returns:
            No-show probability (0-1)
        """
        # Start with base rate
        base_rate = self.base_rates.get(segment, 0.10)
        
        # Adjust for fare class
        fare_multiplier = self.fare_class_multipliers.get(booking_class, 1.0)
        rate = base_rate * fare_multiplier
        
        # Advance purchase adjustment
        # Last-minute bookings have much lower no-show rates
        if advance_purchase_days <= 1:
            rate *= 0.3  # 70% reduction for same-day/next-day
        elif advance_purchase_days <= 7:
            rate *= 0.6  # 40% reduction for within a week
        elif advance_purchase_days <= 21:
            rate *= 0.8  # 20% reduction for within 3 weeks
        
        # Route characteristics
        if route_distance > 2000:  # Long-haul
            rate *= 0.85  # Slightly lower no-show on long flights
        
        if is_hub_flight:
            rate *= 1.2  # Higher no-show on connecting flights
        
        # Cap between reasonable bounds
        return max(0.01, min(0.40, rate))
    
    def estimate_show_up_distribution(
        self,
        bookings: List[Booking],
        capacity: int
    ) -> 'ShowUpForecast':
        """
        Estimate the distribution of passenger show-ups.
        
        Args:
            bookings: List of confirmed bookings
            capacity: Aircraft capacity
            
        Returns:
            ShowUpForecast with mean and distribution
        """
        if not bookings:
            return ShowUpForecast(
                expected_shows=0,
                std_dev=0,
                probability_oversold=0.0,
                expected_denied_boardings=0.0
            )
        
        # Calculate show-up probability for each booking
        show_probs = []
        for booking in bookings:
            customer = booking.customer
            booking_class = booking.solution.booking_classes[0]
            
            # Get no-show probability
            no_show_prob = self.get_no_show_probability(
                segment=customer.segment,
                booking_class=booking_class,
                advance_purchase_days=customer.advance_purchase_days,
                route_distance=1000,  # Placeholder - should come from route
                is_hub_flight=False
            )
            
            show_prob = 1 - no_show_prob
            show_probs.append(show_prob)
        
        # Total bookings
        n_bookings = len(bookings)
        
        # Expected shows (sum of individual probabilities)
        expected_shows = sum(show_probs)
        
        # Variance (assuming independence)
        # Var(X) = sum of p_i * (1 - p_i)
        variance = sum(p * (1 - p) for p in show_probs)
        std_dev = np.sqrt(variance)
        
        # Probability of oversale (shows > capacity)
        if std_dev > 0:
            z_score = (capacity - expected_shows) / std_dev
            prob_oversold = 1 - stats.norm.cdf(z_score)
        else:
            prob_oversold = 1.0 if expected_shows > capacity else 0.0
        
        # Expected denied boardings
        # Using normal approximation: E[max(0, X - c)]
        if std_dev > 0:
            z = (capacity - expected_shows) / std_dev
            expected_db = std_dev * (stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))
        else:
            expected_db = max(0, expected_shows - capacity)
        
        return ShowUpForecast(
            expected_shows=expected_shows,
            std_dev=std_dev,
            probability_oversold=prob_oversold,
            expected_denied_boardings=expected_db,
            total_bookings=n_bookings
        )


@dataclass
class ShowUpForecast:
    """Forecast of passenger show-ups."""
    expected_shows: float
    std_dev: float
    probability_oversold: float
    expected_denied_boardings: float
    total_bookings: int = 0
    
    @property
    def show_up_rate(self) -> float:
        """Average show-up rate."""
        if self.total_bookings > 0:
            return self.expected_shows / self.total_bookings
        return 0.0


@dataclass
class DeniedBoardingCost:
    """
    Cost structure for denied boarding (involuntary denied boarding).
    
    Based on US DOT regulations and airline policies:
    - Compensation depends on delay length and ticket price
    - Additional costs: rebooking, accommodation, goodwill, reputation
    """
    
    # Direct compensation (regulatory)
    compensation_0_1_hour: float = 0.0      # No compensation
    compensation_1_2_hours: float = 400.0   # Domestic: 200% of fare up to $775
    compensation_2_4_hours: float = 800.0   # Domestic: 400% of fare up to $1,550
    compensation_4plus_hours: float = 1350.0  # International
    
    # Additional costs per denied boarding
    rebooking_cost: float = 150.0           # Staff time, systems
    accommodation_cost: float = 200.0       # Hotel if overnight
    meal_voucher_cost: float = 50.0         # Meals during delay
    
    # Intangible costs
    goodwill_cost: float = 300.0            # Customer satisfaction impact
    reputation_cost: float = 200.0          # Brand damage
    
    # Probability distribution of delay lengths
    prob_1_2_hours: float = 0.4
    prob_2_4_hours: float = 0.3
    prob_4plus_hours: float = 0.3
    
    def get_expected_cost(self, avg_fare: float = 500.0) -> float:
        """
        Calculate expected cost per denied boarding.
        
        Args:
            avg_fare: Average ticket price for the flight
            
        Returns:
            Expected total cost per denied boarding
        """
        # Expected regulatory compensation
        compensation = (
            self.prob_1_2_hours * min(self.compensation_1_2_hours, avg_fare * 2) +
            self.prob_2_4_hours * min(self.compensation_2_4_hours, avg_fare * 4) +
            self.prob_4plus_hours * self.compensation_4plus_hours
        )
        
        # Expected operational costs
        operational = (
            self.rebooking_cost +
            self.accommodation_cost * 0.4 +  # Not all delays require hotel
            self.meal_voucher_cost
        )
        
        # Intangible costs
        intangible = self.goodwill_cost + self.reputation_cost
        
        return compensation + operational + intangible


@dataclass
class OverbookingPolicy:
    """Overbooking policy for a flight."""
    booking_limit: int          # Maximum bookings to accept
    capacity: int               # Physical capacity
    overbooking_level: int      # booking_limit - capacity
    expected_shows: float       # Expected number of shows
    expected_revenue_gain: float  # Expected revenue from overbooking
    expected_db_cost: float     # Expected denied boarding cost
    net_benefit: float          # Expected revenue gain - expected cost
    risk_of_denied_boarding: float  # Probability of any DB
    method: OverbookingMethod
    
    @property
    def overbooking_rate(self) -> float:
        """Overbooking as percentage of capacity."""
        if self.capacity > 0:
            return self.overbooking_level / self.capacity
        return 0.0


class OverbookingOptimizer:
    """
    Optimize overbooking limits for flights.
    
    Uses critical fractile policy:
    Accept booking if: p(shows ≤ capacity) ≥ CF
    Where CF = fare / (fare + denied_boarding_cost)
    """
    
    def __init__(
        self,
        no_show_model: Optional[NoShowModel] = None,
        db_cost_model: Optional[DeniedBoardingCost] = None,
        method: OverbookingMethod = OverbookingMethod.CRITICAL_FRACTILE
    ):
        """
        Initialize overbooking optimizer.
        
        Args:
            no_show_model: Model for no-show probabilities
            db_cost_model: Model for denied boarding costs
            method: Optimization method to use
        """
        self.no_show_model = no_show_model or NoShowModel()
        self.db_cost = db_cost_model or DeniedBoardingCost()
        self.method = method
    
    def calculate_overbooking_limit(
        self,
        capacity: int,
        current_bookings: List[Booking],
        avg_fare: float,
        risk_tolerance: float = 0.05
    ) -> OverbookingPolicy:
        """
        Calculate optimal overbooking limit for a flight.
        
        Args:
            capacity: Aircraft capacity (seats)
            current_bookings: Current confirmed bookings
            avg_fare: Average fare for the flight
            risk_tolerance: Maximum acceptable probability of denied boarding
            
        Returns:
            OverbookingPolicy with optimal booking limit
        """
        if self.method == OverbookingMethod.CRITICAL_FRACTILE:
            return self._critical_fractile_method(
                capacity, current_bookings, avg_fare, risk_tolerance
            )
        elif self.method == OverbookingMethod.RISK_AVERSE:
            return self._risk_averse_method(
                capacity, current_bookings, avg_fare, risk_tolerance
            )
        else:
            # Default to critical fractile
            return self._critical_fractile_method(
                capacity, current_bookings, avg_fare, risk_tolerance
            )
    
    def _critical_fractile_method(
        self,
        capacity: int,
        current_bookings: List[Booking],
        avg_fare: float,
        risk_tolerance: float
    ) -> OverbookingPolicy:
        """
        Calculate overbooking limit using critical fractile.
        
        The critical fractile approach:
        - Accept bookings up to the point where the probability of 
          denied boarding exceeds the critical ratio
        - Critical ratio = revenue / (revenue + cost)
        """
        # Get show-up forecast for current bookings
        base_forecast = self.no_show_model.estimate_show_up_distribution(
            current_bookings, capacity
        )
        
        # Expected denied boarding cost
        db_cost = self.db_cost.get_expected_cost(avg_fare)
        
        # Critical fractile
        cf = avg_fare / (avg_fare + db_cost)
        
        # Find optimal booking limit
        # We want: P(shows ≤ capacity) ≥ CF
        # Or equivalently: P(shows > capacity) ≤ 1 - CF
        
        max_risk = min(risk_tolerance, 1 - cf)
        
        # Search for optimal booking limit
        best_limit = capacity
        best_net_benefit = 0.0
        
        # Try different booking limits
        for additional_bookings in range(0, int(capacity * 0.3) + 1):  # Up to 30% overbook
            trial_limit = capacity + additional_bookings
            
            # Estimate shows if we had this many bookings
            # Simplified: assume new bookings have similar characteristics
            avg_show_rate = base_forecast.show_up_rate if base_forecast.show_up_rate > 0 else 0.90
            
            expected_shows = trial_limit * avg_show_rate
            std_dev = np.sqrt(trial_limit * avg_show_rate * (1 - avg_show_rate))
            
            if std_dev > 0:
                z_score = (capacity - expected_shows) / std_dev
                prob_oversold = 1 - stats.norm.cdf(z_score)
                
                # Expected denied boardings
                z = (capacity - expected_shows) / std_dev
                expected_db = std_dev * (stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))
            else:
                prob_oversold = 1.0 if expected_shows > capacity else 0.0
                expected_db = max(0, expected_shows - capacity)
            
            # Skip if risk is too high
            if prob_oversold > max_risk:
                break
            
            # Calculate net benefit
            revenue_gain = additional_bookings * avg_fare * avg_show_rate
            expected_cost = expected_db * db_cost
            net_benefit = revenue_gain - expected_cost
            
            if net_benefit > best_net_benefit:
                best_net_benefit = net_benefit
                best_limit = trial_limit
        
        # Final forecast with best limit
        avg_show_rate = base_forecast.show_up_rate if base_forecast.show_up_rate > 0 else 0.90
        expected_shows = best_limit * avg_show_rate
        std_dev = np.sqrt(best_limit * avg_show_rate * (1 - avg_show_rate))
        
        if std_dev > 0:
            z_score = (capacity - expected_shows) / std_dev
            prob_oversold = 1 - stats.norm.cdf(z_score)
            z = (capacity - expected_shows) / std_dev
            expected_db = std_dev * (stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))
        else:
            prob_oversold = 1.0 if expected_shows > capacity else 0.0
            expected_db = max(0, expected_shows - capacity)
        
        overbooking_level = best_limit - capacity
        revenue_gain = overbooking_level * avg_fare * avg_show_rate
        db_cost_expected = expected_db * db_cost
        
        return OverbookingPolicy(
            booking_limit=best_limit,
            capacity=capacity,
            overbooking_level=overbooking_level,
            expected_shows=expected_shows,
            expected_revenue_gain=revenue_gain,
            expected_db_cost=db_cost_expected,
            net_benefit=best_net_benefit,
            risk_of_denied_boarding=prob_oversold,
            method=self.method
        )
    
    def _risk_averse_method(
        self,
        capacity: int,
        current_bookings: List[Booking],
        avg_fare: float,
        risk_tolerance: float
    ) -> OverbookingPolicy:
        """
        Conservative overbooking - minimize denied boarding risk.
        
        Only overbook if risk is very low.
        """
        # Use half the normal risk tolerance
        conservative_risk = risk_tolerance * 0.5
        
        base_forecast = self.no_show_model.estimate_show_up_distribution(
            current_bookings, capacity
        )
        
        avg_show_rate = base_forecast.show_up_rate if base_forecast.show_up_rate > 0 else 0.90
        
        # Conservative: only overbook by expected no-shows
        expected_no_shows = len(current_bookings) * (1 - avg_show_rate)
        overbooking_level = max(0, int(expected_no_shows * 0.7))  # 70% of expected no-shows
        
        best_limit = capacity + overbooking_level
        
        expected_shows = best_limit * avg_show_rate
        std_dev = np.sqrt(best_limit * avg_show_rate * (1 - avg_show_rate))
        
        if std_dev > 0:
            z_score = (capacity - expected_shows) / std_dev
            prob_oversold = 1 - stats.norm.cdf(z_score)
            z = (capacity - expected_shows) / std_dev
            expected_db = std_dev * (stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z)))
        else:
            prob_oversold = 0.0
            expected_db = 0.0
        
        db_cost = self.db_cost.get_expected_cost(avg_fare)
        revenue_gain = overbooking_level * avg_fare * avg_show_rate
        db_cost_expected = expected_db * db_cost
        
        return OverbookingPolicy(
            booking_limit=best_limit,
            capacity=capacity,
            overbooking_level=overbooking_level,
            expected_shows=expected_shows,
            expected_revenue_gain=revenue_gain,
            expected_db_cost=db_cost_expected,
            net_benefit=revenue_gain - db_cost_expected,
            risk_of_denied_boarding=prob_oversold,
            method=self.method
        )
    
    def should_accept_booking(
        self,
        flight: FlightDate,
        booking_class: BookingClass,
        party_size: int,
        overbooking_policy: OverbookingPolicy
    ) -> bool:
        """
        Determine if a booking should be accepted given overbooking policy.
        
        Args:
            flight: Flight to book
            booking_class: Requested booking class
            party_size: Number of passengers
            overbooking_policy: Current overbooking policy
            
        Returns:
            True if booking should be accepted
        """
        # Total current bookings
        total_bookings = sum(flight.bookings.values())
        
        # Check against booking limit
        if total_bookings + party_size <= overbooking_policy.booking_limit:
            return True
        
        return False
    
    def simulate_denied_boarding(
        self,
        bookings: List[Booking],
        capacity: int,
        rng: np.random.Generator
    ) -> Tuple[int, List[Booking]]:
        """
        Simulate show-ups and determine denied boardings.
        
        Args:
            bookings: List of confirmed bookings
            capacity: Aircraft capacity
            rng: Random number generator
            
        Returns:
            Tuple of (number_of_shows, list_of_denied_bookings)
        """
        # Simulate show-up for each booking
        shows = []
        no_shows = []
        
        for booking in bookings:
            customer = booking.customer
            booking_class = booking.solution.booking_classes[0]
            
            # Get no-show probability
            no_show_prob = self.no_show_model.get_no_show_probability(
                segment=customer.segment,
                booking_class=booking_class,
                advance_purchase_days=customer.advance_purchase_days,
                route_distance=1000,
                is_hub_flight=False
            )
            
            # Simulate
            if rng.random() > no_show_prob:
                shows.append(booking)
            else:
                no_shows.append(booking)
        
        # If shows exceed capacity, deny boarding to lowest fare pax
        denied = []
        if len(shows) > capacity:
            # Sort by fare (lowest first for denial)
            shows_sorted = sorted(shows, key=lambda b: b.solution.total_price)
            denied = shows_sorted[capacity:]
            shows = shows_sorted[:capacity]
        
        return len(shows), denied
