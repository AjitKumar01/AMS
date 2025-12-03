"""
Network Revenue Management (O-D Control) implementation.

Key concepts:
- Virtual nesting: Aggregate booking classes across flights
- Displacement costs: Opportunity cost of accepting a booking
- Network optimization: Optimize across entire network, not just legs
- Bid prices: Shadow prices for inventory allocation
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import date
import numpy as np
from scipy.optimize import linprog

from core.models import (
    FlightDate, Route, Fare, BookingClass, 
    TravelSolution, Airport
)


@dataclass
class VirtualBucket:
    """
    Virtual bucket for network RM.
    
    Aggregates booking classes across the network based on revenue value,
    not physical cabin/class.
    """
    bucket_id: int
    min_revenue: float
    max_revenue: float
    protection_level: int = 0
    authorization_level: int = 0
    
    def contains_fare(self, fare: Fare) -> bool:
        """Check if fare belongs to this bucket."""
        return self.min_revenue <= fare.base_amount <= self.max_revenue


@dataclass
class DisplacementCost:
    """
    Displacement cost for a flight leg.
    
    Represents the expected revenue loss from accepting a booking
    that consumes inventory.
    """
    flight_date_key: str
    bid_price: float  # Shadow price per seat
    remaining_capacity: int
    expected_future_demand: float
    last_updated: date


class NetworkOptimizer:
    """
    Network Revenue Management optimizer.
    
    Implements O-D control with:
    - Virtual nesting
    - Displacement cost calculation
    - Network-wide optimization
    - Bid price computation
    """
    
    def __init__(
        self,
        num_virtual_buckets: int = 10,
        optimization_method: str = "linear_programming"
    ):
        """
        Initialize network optimizer.
        
        Args:
            num_virtual_buckets: Number of virtual buckets to create
            optimization_method: "linear_programming" or "approximate"
        """
        self.num_virtual_buckets = num_virtual_buckets
        self.optimization_method = optimization_method
        
        self.virtual_buckets: List[VirtualBucket] = []
        self.displacement_costs: Dict[str, DisplacementCost] = {}
        self.bid_prices: Dict[str, float] = {}
        
        # Network structure
        self.itinerary_to_legs: Dict[str, List[str]] = {}  # itinerary_id -> leg_ids
        self.leg_to_itineraries: Dict[str, List[str]] = {}  # leg_id -> itinerary_ids
    
    def create_virtual_buckets(self, min_fare: float, max_fare: float):
        """
        Create virtual buckets spanning the fare range.
        
        Args:
            min_fare: Minimum fare in the network
            max_fare: Maximum fare in the network
        """
        self.virtual_buckets = []
        fare_range = max_fare - min_fare
        bucket_size = fare_range / self.num_virtual_buckets
        
        for i in range(self.num_virtual_buckets):
            bucket_min = min_fare + i * bucket_size
            bucket_max = min_fare + (i + 1) * bucket_size
            
            self.virtual_buckets.append(VirtualBucket(
                bucket_id=i,
                min_revenue=bucket_min,
                max_revenue=bucket_max
            ))
    
    def register_itinerary(self, itinerary_id: str, leg_ids: List[str]):
        """
        Register an itinerary and its component legs.
        
        Args:
            itinerary_id: Unique itinerary identifier (e.g., "JFK-ORD-LAX")
            leg_ids: List of flight leg identifiers
        """
        self.itinerary_to_legs[itinerary_id] = leg_ids
        
        for leg_id in leg_ids:
            if leg_id not in self.leg_to_itineraries:
                self.leg_to_itineraries[leg_id] = []
            self.leg_to_itineraries[leg_id].append(itinerary_id)
    
    def calculate_displacement_cost(
        self,
        flight_date: FlightDate,
        forecasted_demand: Dict[BookingClass, float],
        fares: Dict[BookingClass, Fare]
    ) -> float:
        """
        Calculate displacement cost (bid price) for a flight leg.
        
        Uses expected marginal seat revenue (EMSR) approach.
        
        Args:
            flight_date: Flight to calculate for
            forecasted_demand: Demand forecast by booking class
            fares: Fares by booking class
        
        Returns:
            Bid price (displacement cost) per seat
        """
        # Get remaining capacity
        remaining = flight_date.get_available_seats()
        
        if remaining <= 0:
            return float('inf')  # No capacity, infinite cost
        
        # Sort fare classes by revenue (descending)
        fare_classes = sorted(
            fares.items(),
            key=lambda x: x[1].base_amount,
            reverse=True
        )
        
        # Calculate expected revenue for next seat
        # This is a simplified EMSR-style calculation
        expected_revenues = []
        
        for booking_class, fare in fare_classes:
            demand = forecasted_demand.get(booking_class, 0.0)
            revenue = fare.base_amount
            
            # Probability of selling to this class
            prob_sell = min(1.0, demand / max(remaining, 1))
            
            expected_revenues.append(prob_sell * revenue)
        
        # Bid price is the expected revenue of the next seat
        bid_price = sum(expected_revenues)
        
        # Store displacement cost
        self.displacement_costs[flight_date.flight_id] = DisplacementCost(
            flight_date_key=flight_date.flight_id,
            bid_price=bid_price,
            remaining_capacity=remaining,
            expected_future_demand=sum(forecasted_demand.values()),
            last_updated=date.today()
        )
        
        self.bid_prices[flight_date.flight_id] = bid_price
        
        return bid_price
    
    def calculate_itinerary_value(
        self,
        solution: TravelSolution,
        fare: float
    ) -> float:
        """
        Calculate network value of accepting an itinerary.
        
        Value = Fare - Sum of displacement costs for all legs
        
        Args:
            solution: Travel solution (may include connections)
            fare: Offered fare
        
        Returns:
            Network value (can be negative if displacement cost > fare)
        """
        total_displacement = 0.0
        
        for flight in solution.flights:
            leg_key = flight.flight_id
            
            if leg_key in self.bid_prices:
                # Displacement cost for this leg
                total_displacement += self.bid_prices[leg_key]
            else:
                # No bid price available, assume zero displacement
                total_displacement += 0.0
        
        # Network value
        network_value = fare - total_displacement
        
        return network_value
    
    def should_accept_booking(
        self,
        solution: TravelSolution,
        fare: float,
        party_size: int
    ) -> bool:
        """
        Decide whether to accept a booking based on network value.
        
        Args:
            solution: Proposed travel solution
            fare: Total fare for party
            party_size: Number of passengers
        
        Returns:
            True if booking should be accepted
        """
        # Calculate per-passenger network value
        per_pax_fare = fare / party_size
        network_value = self.calculate_itinerary_value(solution, per_pax_fare)
        
        # Accept if network value is positive
        return network_value > 0
    
    def optimize_network(
        self,
        flight_dates: Dict[str, FlightDate],
        demand_forecasts: Dict[str, Dict[BookingClass, float]],
        itineraries: List[Tuple[str, List[str], float, float]]  # (id, legs, fare, demand)
    ) -> Dict[str, Dict[BookingClass, int]]:
        """
        Optimize booking limits across the network using linear programming.
        
        This solves the network RM problem:
        Maximize: Sum of (fare * bookings) for all itineraries
        Subject to:
        - Capacity constraints on each leg
        - Demand constraints on each itinerary
        
        Args:
            flight_dates: All flight dates in network
            demand_forecasts: Demand forecasts by flight and class
            itineraries: List of (itinerary_id, leg_ids, fare, demand)
        
        Returns:
            Optimal booking limits by flight and class
        """
        if self.optimization_method == "linear_programming":
            return self._optimize_network_lp(
                flight_dates, demand_forecasts, itineraries
            )
        else:
            return self._optimize_network_approximate(
                flight_dates, demand_forecasts, itineraries
            )
    
    def _optimize_network_lp(
        self,
        flight_dates: Dict[str, FlightDate],
        demand_forecasts: Dict[str, Dict[BookingClass, float]],
        itineraries: List[Tuple[str, List[str], float, float]]
    ) -> Dict[str, Dict[BookingClass, int]]:
        """
        Linear programming network optimization.
        
        Decision variables: Number of bookings to accept for each itinerary
        Objective: Maximize total revenue
        Constraints: Leg capacities, demand limits
        """
        n_itineraries = len(itineraries)
        
        if n_itineraries == 0:
            return {}
        
        # Objective: maximize revenue (negate for minimization)
        c = [-itin[2] for itin in itineraries]  # -fare for each itinerary
        
        # Bounds: 0 <= bookings <= demand
        bounds = [(0, itin[3]) for itin in itineraries]
        
        # Constraints: capacity on each leg
        # A_ub @ x <= b_ub
        A_ub = []
        b_ub = []
        
        for leg_id, fd in flight_dates.items():
            # Create constraint row for this leg
            constraint_row = []
            
            for itin_id, leg_ids, fare, demand in itineraries:
                # Does this itinerary use this leg?
                if leg_id in leg_ids:
                    constraint_row.append(1)  # Uses 1 seat
                else:
                    constraint_row.append(0)  # Doesn't use this leg
            
            A_ub.append(constraint_row)
            b_ub.append(fd.schedule.aircraft.total_capacity)
        
        # Solve LP
        try:
            result = linprog(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                bounds=bounds,
                method='highs'
            )
            
            if result.success:
                # Extract booking limits
                # Map back to flight/class structure
                booking_limits = {}
                
                for i, (itin_id, leg_ids, fare, demand) in enumerate(itineraries):
                    optimal_bookings = result.x[i]
                    
                    # Allocate to legs (simplified - equal split)
                    for leg_id in leg_ids:
                        if leg_id not in booking_limits:
                            booking_limits[leg_id] = {}
                        
                        # Map to booking class based on fare
                        booking_class = self._fare_to_class(fare)
                        
                        if booking_class not in booking_limits[leg_id]:
                            booking_limits[leg_id][booking_class] = 0
                        
                        booking_limits[leg_id][booking_class] += int(optimal_bookings)
                
                return booking_limits
            else:
                # Optimization failed, fall back to approximate
                return self._optimize_network_approximate(
                    flight_dates, demand_forecasts, itineraries
                )
        
        except Exception as e:
            # LP failed, use approximate method
            return self._optimize_network_approximate(
                flight_dates, demand_forecasts, itineraries
            )
    
    def _optimize_network_approximate(
        self,
        flight_dates: Dict[str, FlightDate],
        demand_forecasts: Dict[str, Dict[BookingClass, float]],
        itineraries: List[Tuple[str, List[str], float, float]]
    ) -> Dict[str, Dict[BookingClass, int]]:
        """
        Approximate network optimization using bid prices.
        
        Simpler heuristic approach when LP is not feasible.
        """
        booking_limits = {}
        
        # Calculate bid prices for all legs
        for leg_id, fd in flight_dates.items():
            forecast = demand_forecasts.get(leg_id, {})
            fares_dict = fd.fares if hasattr(fd, 'fares') else {}
            
            bid_price = self.calculate_displacement_cost(fd, forecast, fares_dict)
            
            # Allocate capacity to itineraries with value > bid price
            if leg_id not in booking_limits:
                booking_limits[leg_id] = {}
            
            remaining_capacity = fd.schedule.aircraft.total_capacity
            
            # Sort itineraries by revenue (descending)
            sorted_itins = sorted(
                [it for it in itineraries if leg_id in it[1]],
                key=lambda x: x[2],
                reverse=True
            )
            
            for itin_id, leg_ids, fare, demand in sorted_itins:
                if fare > bid_price and remaining_capacity > 0:
                    # Accept up to demand or remaining capacity
                    accept = min(int(demand), remaining_capacity)
                    
                    booking_class = self._fare_to_class(fare)
                    
                    if booking_class not in booking_limits[leg_id]:
                        booking_limits[leg_id][booking_class] = 0
                    
                    booking_limits[leg_id][booking_class] += accept
                    remaining_capacity -= accept
        
        return booking_limits
    
    def _fare_to_class(self, fare: float) -> BookingClass:
        """Map fare amount to booking class (simplified)."""
        if fare >= 800:
            return BookingClass.F
        elif fare >= 500:
            return BookingClass.J
        elif fare >= 300:
            return BookingClass.W
        elif fare >= 200:
            return BookingClass.Y
        else:
            return BookingClass.L
    
    def get_displacement_report(self) -> Dict:
        """Generate report on displacement costs across network."""
        report = {
            'total_legs': len(self.displacement_costs),
            'avg_bid_price': 0.0,
            'max_bid_price': 0.0,
            'min_bid_price': float('inf'),
            'legs': {}
        }
        
        if not self.displacement_costs:
            return report
        
        bid_prices = [dc.bid_price for dc in self.displacement_costs.values()]
        
        report['avg_bid_price'] = np.mean(bid_prices)
        report['max_bid_price'] = max(bid_prices)
        report['min_bid_price'] = min(bid_prices)
        
        # Per-leg details
        for leg_id, dc in self.displacement_costs.items():
            report['legs'][leg_id] = {
                'bid_price': dc.bid_price,
                'remaining_capacity': dc.remaining_capacity,
                'expected_demand': dc.expected_future_demand,
                'utilization': 1.0 - (dc.remaining_capacity / 
                               (dc.remaining_capacity + dc.expected_future_demand))
                               if dc.remaining_capacity + dc.expected_future_demand > 0 else 0.0
            }
        
        return report
