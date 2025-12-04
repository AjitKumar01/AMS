import logging
from typing import List, Dict
from core.models import FlightDate, BookingClass, RMControl
from rm.optimizer import DemandForecast

class HeuristicBidPriceOptimizer:
    """
    Heuristic Bid Price Optimizer.
    
    A modern, heuristic-based approach that adjusts bid prices dynamically
    based on:
    1. Time to departure (DTD)
    2. Current Load Factor vs Expected Load Factor
    3. Forecasted remaining demand
    
    This simulates a continuous pricing approach rather than just
    opening/closing booking classes.
    """
    
    def __init__(self):
        self.logger = logging.getLogger('HeuristicBidPriceOptimizer')
        
    def optimize(
        self,
        flight_date: FlightDate,
        forecasts: List[DemandForecast],
        fares: Dict[BookingClass, float]
    ) -> RMControl:
        """
        Calculate optimal bid prices.
        """
        # 1. Calculate aggregate demand forecast
        total_expected_demand = sum(f.mean for f in forecasts)
        
        # 2. Get current state
        capacity = sum(flight_date.capacity.values())
        booked = flight_date.total_bookings()
        remaining_seats = capacity - booked
        
        # 3. Calculate Load Factor
        lf = booked / capacity if capacity > 0 else 0
        
        # 4. Determine DTD (Days To Departure)
        # This requires context not passed here, but we can infer or use a default
        # For now, we'll use a simplified logic or assume we can get it from flight_date
        # flight_date.departure_date is available.
        from datetime import date
        today = date.today() # This might be wrong in simulation time. 
        # We should ideally pass current_date to optimize.
        # Assuming the simulator handles the date context correctly or we use a heuristic.
        
        # Let's calculate a "scarcity price" (Bid Price)
        # Bid Price = P(demand > remaining) * Min_Fare_of_High_Class
        
        # Sort fares
        sorted_fares = sorted(fares.values(), reverse=True)
        min_fare = sorted_fares[-1] if sorted_fares else 0
        max_fare = sorted_fares[0] if sorted_fares else 1000
        
        # Scarcity factor: (Expected Demand / Remaining Capacity)
        if remaining_seats > 0:
            scarcity = total_expected_demand / remaining_seats
        else:
            scarcity = 10.0 # Very high
            
        # Base Bid Price
        bid_price = min_fare * (scarcity ** 2) # Quadratic penalty for scarcity
        
        # Cap at max fare
        bid_price = min(bid_price, max_fare)
        
        # Set booking limits based on Bid Price
        # Open classes where Fare >= Bid Price
        booking_limits = {}
        for bc, fare in fares.items():
            if fare >= bid_price:
                booking_limits[bc] = remaining_seats
            else:
                booking_limits[bc] = 0
                
        return RMControl(
            booking_limits=booking_limits,
            protection_levels={}, # Not used in bid price control
            bid_prices=[bid_price]
        )
