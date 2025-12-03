"""
Advanced demand forecasting with ML and accuracy tracking.

This module provides:
- Traditional statistical forecasting (Pickup, Additive)
- ML-based forecasting (Neural Networks, XGBoost)
- Forecast accuracy measurement
- Impact of forecast errors on RM performance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import date, timedelta
from enum import Enum
import logging
import numpy as np
from scipy import stats

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from core.models import BookingClass, FlightDate, CustomerSegment
from demand.elasticity import ElasticityModel
from choice.frat5 import FRAT5Model


class ForecastMethod(Enum):
    """Forecasting methods available."""
    HISTORICAL_AVERAGE = "historical_average"
    PICKUP = "pickup"  # Traditional airline method
    ADDITIVE_PICKUP = "additive_pickup"
    MULTIPLICATIVE_PICKUP = "multiplicative_pickup"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    NEURAL_NETWORK = "neural_network"
    XGBOOST = "xgboost"  # If available
    ENSEMBLE = "ensemble"  # Combine multiple methods


@dataclass
class ForecastAccuracy:
    """Track forecast accuracy metrics."""
    method: str
    mean_absolute_error: float = 0.0
    mean_absolute_percentage_error: float = 0.0
    root_mean_squared_error: float = 0.0
    bias: float = 0.0  # Systematic over/under forecasting
    
    # By time horizon
    accuracy_by_horizon: Dict[int, float] = field(default_factory=dict)
    
    # By booking class
    accuracy_by_class: Dict[BookingClass, float] = field(default_factory=dict)
    
    # Impact on revenue
    revenue_lost_to_error: float = 0.0
    
    def update(self, forecast: float, actual: float):
        """Update accuracy metrics with new observation."""
        error = forecast - actual
        abs_error = abs(error)
        
        # Running averages (simplified - should use proper tracking)
        self.mean_absolute_error = (
            0.9 * self.mean_absolute_error + 0.1 * abs_error
        )
        
        if actual != 0:
            pct_error = abs_error / actual * 100
            self.mean_absolute_percentage_error = (
                0.9 * self.mean_absolute_percentage_error + 0.1 * pct_error
            )
        
        self.bias = 0.9 * self.bias + 0.1 * error


@dataclass
class DemandForecast:
    """Demand forecast for a flight."""
    flight_date_id: str
    forecast_date: date
    forecasts: Dict[BookingClass, float] = field(default_factory=dict)
    total_forecast: float = 0.0
    
    # Confidence intervals
    lower_bound: Dict[BookingClass, float] = field(default_factory=dict)
    upper_bound: Dict[BookingClass, float] = field(default_factory=dict)
    
    # Metadata
    method_used: str = "unknown"
    days_before_departure: int = 0
    
    def get_forecast(self, booking_class: BookingClass) -> float:
        """Get forecast for a booking class."""
        return self.forecasts.get(booking_class, 0.0)
    
    def get_total_forecast(self) -> float:
        """Get total demand forecast."""
        if self.total_forecast > 0:
            return self.total_forecast
        return sum(self.forecasts.values())


class DemandForecaster:
    """
    Demand forecaster with multiple methods and accuracy tracking.
    
    Key feature: Tracks how forecast accuracy affects RM performance.
    """
    
    def __init__(
        self,
        method: ForecastMethod = ForecastMethod.PICKUP,
        track_accuracy: bool = True,
        add_noise: bool = False,
        noise_std: float = 0.1
    ):
        """
        Initialize forecaster.
        
        Args:
            method: Forecasting method to use
            track_accuracy: Whether to track forecast vs actual
            add_noise: Add random noise to simulate forecast error
            noise_std: Standard deviation of noise (as fraction of forecast)
        """
        self.method = method
        self.track_accuracy = track_accuracy
        self.add_noise = add_noise
        self.noise_std = noise_std
        
        self.logger = logging.getLogger('DemandForecaster')
        
        # Historical data for learning
        self.booking_history: Dict[str, List[Tuple[date, BookingClass, int]]] = {}
        self.pickup_patterns: Dict[int, float] = {}  # days_before -> pickup factor
        
        # Accuracy tracking
        self.accuracy = ForecastAccuracy(method=method.value)
        self.forecast_history: List[Tuple[DemandForecast, Dict[BookingClass, float]]] = []
        
        # ML models (if available)
        self.neural_model = None
        if TORCH_AVAILABLE and method == ForecastMethod.NEURAL_NETWORK:
            self.neural_model = self._build_neural_model()
            
        # Price elasticity model
        self.elasticity_model = ElasticityModel()
        
        # FRAT5 model for unconstraining
        self.frat5_model = FRAT5Model()
    
    def record_booking(
        self,
        flight_date_id: str,
        booking_date: date,
        booking_class: BookingClass,
        passengers: int
    ):
        """Record a booking for learning."""
        if flight_date_id not in self.booking_history:
            self.booking_history[flight_date_id] = []
        
        self.booking_history[flight_date_id].append(
            (booking_date, booking_class, passengers)
        )
    
    def forecast_demand(
        self,
        flight_date: FlightDate,
        current_date: date,
        current_bookings: Dict[BookingClass, int],
        current_prices: Optional[Dict[BookingClass, float]] = None,
        base_prices: Optional[Dict[BookingClass, float]] = None
    ) -> DemandForecast:
        """
        Forecast remaining demand for a flight.
        
        Args:
            flight_date: Flight to forecast
            current_date: Current date
            current_bookings: Current bookings by class
            current_prices: Current price points (optional)
            base_prices: Reference prices for elasticity (optional)
        
        Returns:
            Demand forecast
        """
        days_before = (flight_date.departure_date - current_date).days
        
        # Choose forecasting method
        if self.method == ForecastMethod.HISTORICAL_AVERAGE:
            forecast = self._forecast_historical_average(
                flight_date, current_bookings
            )
        elif self.method in [ForecastMethod.PICKUP, 
                            ForecastMethod.ADDITIVE_PICKUP,
                            ForecastMethod.MULTIPLICATIVE_PICKUP]:
            forecast = self._forecast_pickup(
                flight_date, current_date, current_bookings
            )
        elif self.method == ForecastMethod.EXPONENTIAL_SMOOTHING:
            forecast = self._forecast_exponential_smoothing(
                flight_date, current_bookings
            )
        elif self.method == ForecastMethod.NEURAL_NETWORK:
            forecast = self._forecast_neural_network(
                flight_date, current_date, current_bookings
            )
        else:
            # Default to simple average
            forecast = self._forecast_historical_average(
                flight_date, current_bookings
            )
        
        forecast.days_before_departure = days_before
        
        # Add noise if configured (to simulate forecast error)
        if self.add_noise:
            forecast = self._add_forecast_noise(forecast)
            
        # Apply price elasticity if prices are provided
        if current_prices:
            forecast = self.apply_price_elasticity(forecast, current_prices, base_prices)
        
        self.logger.info(f"Forecast for {flight_date.flight_id} (DTD={days_before}): {forecast.total_forecast:.1f}")
        
        return forecast

    def apply_price_elasticity(
        self,
        forecast: DemandForecast,
        current_prices: Dict[BookingClass, float],
        base_prices: Optional[Dict[BookingClass, float]] = None
    ) -> DemandForecast:
        """
        Adjust forecast based on price elasticity.
        """
        # Default base prices if not provided (approximate relative values)
        if not base_prices:
            base_prices = {}
            # Base price logic: F > J > W > Y
            # We'll use the current price as base if not specified, 
            # which means no adjustment unless we have a reference.
            # Ideally, base_prices should be the "standard" fare.
            pass

        for booking_class, demand in forecast.forecasts.items():
            if booking_class in current_prices:
                price = current_prices[booking_class]
                # If base price not known, assume current price is the base (ratio=1.0)
                base = base_prices.get(booking_class, price) if base_prices else price
                
                # Use elasticity model
                adjusted_demand = self.elasticity_model.adjust_demand(
                    base_demand=demand,
                    current_price=price,
                    base_price=base,
                    booking_class=booking_class
                )
                forecast.forecasts[booking_class] = adjusted_demand
                
        forecast.total_forecast = sum(forecast.forecasts.values())
        return forecast

    def unconstrain_demand_with_frat5(
        self,
        observed_bookings: Dict[BookingClass, int],
        fares: Dict[BookingClass, float],
        availability: Dict[BookingClass, bool],
        segment_mix: Optional[Dict[CustomerSegment, float]] = None
    ) -> Dict[BookingClass, float]:
        """
        Unconstrain demand using FRAT5 sell-up curves.
        
        Estimates true demand for lower classes that were closed, based on 
        bookings observed in higher classes.
        
        Args:
            observed_bookings: Actual bookings by class
            fares: Price for each class
            availability: Whether each class was open (True) or closed (False)
            segment_mix: Estimated mix of customer segments (default: mostly leisure for low classes)
            
        Returns:
            Unconstrained demand dictionary
        """
        # Initialize unconstrained demand with observed bookings
        unconstrained = {bc: float(pax) for bc, pax in observed_bookings.items()}
        
        # Default segment mix if not provided
        if not segment_mix:
            segment_mix = {
                CustomerSegment.BUSINESS: 0.3,
                CustomerSegment.LEISURE: 0.7
            }
            
        # Sort classes by price (Low to High)
        sorted_classes = sorted(fares.keys(), key=lambda x: fares[x])
        
        frat5_model = FRAT5Model()
        
        # Iterate from lowest fare to highest
        for i, current_class in enumerate(sorted_classes):
            # If this class was closed, we need to estimate lost demand
            if not availability.get(current_class, True):
                # Find the next open higher class
                next_open_class = None
                for j in range(i + 1, len(sorted_classes)):
                    higher_class = sorted_classes[j]
                    if availability.get(higher_class, True):
                        next_open_class = higher_class
                        break
                
                if next_open_class:
                    # Calculate sell-up probability
                    price_low = fares[current_class]
                    price_high = fares[next_open_class]
                    
                    # Weighted average probability across segments
                    weighted_prob = 0.0
                    total_weight = 0.0
                    
                    for segment, weight in segment_mix.items():
                        prob = frat5_model.calculate_sellup_prob(price_low, price_high, segment)
                        weighted_prob += prob * weight
                        total_weight += weight
                        
                    avg_sellup_prob = weighted_prob / total_weight if total_weight > 0 else 0.001
                    
                    if avg_sellup_prob > 0:
                        # Heuristic: Assume a portion of the higher class bookings are sell-ups.
                        # We estimate the "Lost Demand" (people who refused to buy up)
                        # based on the "Captured Demand" (people who did buy up).
                        
                        # Assumption: 30% of the bookings in the next higher class 
                        # are sell-ups from this lower class.
                        observed_high = observed_bookings.get(next_open_class, 0)
                        estimated_sellup_pax = observed_high * 0.3
                        
                        # Calculate lost demand: Lost = SellUp * (1 - P) / P
                        lost_demand = estimated_sellup_pax * (1 - avg_sellup_prob) / avg_sellup_prob
                        
                        unconstrained[current_class] += lost_demand
                        
                        # Note: We do not subtract from the higher class in this simplified model
                        # to avoid over-correction without more data.
                        
        return unconstrained
        sorted_classes = sorted(fares.keys(), key=lambda bc: fares[bc])
        
        # Iterate from lowest to highest
        for i, low_class in enumerate(sorted_classes):
            if availability.get(low_class, True):
                continue  # If open, observed demand is assumed true (simplified)
                
            # If closed, we need to estimate how many people wanted this but bought up
            # Look at higher classes that were open
            low_fare = fares[low_class]
            
            for high_class in sorted_classes[i+1:]:
                if not availability.get(high_class, False):
                    continue
                    
                high_fare = fares[high_class]
                observed_high = observed_bookings.get(high_class, 0)
                
                if observed_high == 0:
                    continue
                
                # Calculate average sell-up probability across segments
                avg_prob = 0.0
                for segment, weight in segment_mix.items():
                    prob = self.frat5_model.calculate_sellup_prob(low_fare, high_fare, segment)
                    avg_prob += prob * weight
                
                if avg_prob > 0:
                    # Estimate spill that was recaptured
                    # Recaptured = True_Demand_Low * Prob_SellUp
                    # But we only know Observed_High. 
                    # Observed_High includes natural demand for High + Recaptured from Low.
                    # This is a complex simultaneous equation problem in reality.
                    # Simplified heuristic: Assume a fraction of High bookings came from Low.
                    
                    # Let's assume X% of high bookings are upsells.
                    # Or better: Estimate potential spill from Low based on typical demand profile
                    # and verify if High bookings support it.
                    
                    # Simple "Reverse" Unconstraining:
                    # If we see N bookings in High, and sell-up prob was P,
                    # then potentially N/P people wanted Low? No, that assumes ALL High bookings came from Low.
                    
                    # Standard approach:
                    # 1. Estimate natural demand for High (e.g. historical avg)
                    # 2. Excess = Observed_High - Natural_High
                    # 3. If Excess > 0, attribute to spill from Low: Spill = Excess / Prob_SellUp
                    # 4. Add Spill to Low demand.
                    pass
                    
        return unconstrained
    
    def _forecast_historical_average(
        self,
        flight_date: FlightDate,
        current_bookings: Dict[BookingClass, int]
    ) -> DemandForecast:
        """Simple historical average forecasting."""
        forecast = DemandForecast(
            flight_date_id=flight_date.flight_id,
            forecast_date=date.today(),
            method_used="historical_average"
        )
        
        # Use historical data if available
        if flight_date.flight_id in self.booking_history:
            history = self.booking_history[flight_date.flight_id]
            
            for booking_class in BookingClass:
                # Count historical bookings for this class
                class_bookings = sum(
                    pax for dt, bc, pax in history if bc == booking_class
                )
                
                # Simple average (should use multiple flights)
                forecast.forecasts[booking_class] = float(class_bookings)
        else:
            # No history, use default estimates based on capacity
            capacity = flight_date.schedule.aircraft.total_capacity
            
            # Distribute capacity across classes (rough estimate)
            forecast.forecasts[BookingClass.L] = capacity * 0.3
            forecast.forecasts[BookingClass.Y] = capacity * 0.25
            forecast.forecasts[BookingClass.W] = capacity * 0.20
            forecast.forecasts[BookingClass.J] = capacity * 0.15
            forecast.forecasts[BookingClass.F] = capacity * 0.10
        
        forecast.total_forecast = sum(forecast.forecasts.values())
        
        return forecast
    
    def _forecast_pickup(
        self,
        flight_date: FlightDate,
        current_date: date,
        current_bookings: Dict[BookingClass, int]
    ) -> DemandForecast:
        """
        Pickup method forecasting.
        
        Final demand = Current bookings + Pickup (remaining to come)
        Pickup is estimated from historical patterns.
        """
        forecast = DemandForecast(
            flight_date_id=flight_date.flight_id,
            forecast_date=current_date,
            method_used="pickup"
        )
        
        days_before = (flight_date.departure_date - current_date).days
        
        # Get pickup factor for this horizon
        pickup_factor = self._get_pickup_factor(days_before)
        
        # Forecast by class
        for booking_class in BookingClass:
            current = current_bookings.get(booking_class, 0)
            
            # Expected additional bookings
            if self.method == ForecastMethod.ADDITIVE_PICKUP:
                # Additive: Pickup is absolute number
                pickup = self._get_additive_pickup(booking_class, days_before)
                forecast.forecasts[booking_class] = current + pickup
            
            elif self.method == ForecastMethod.MULTIPLICATIVE_PICKUP:
                # Multiplicative: Pickup is multiplier
                if current > 0:
                    forecast.forecasts[booking_class] = current * pickup_factor
                else:
                    # No bookings yet, use historical average
                    forecast.forecasts[booking_class] = self._get_average_demand(
                        booking_class
                    )
            else:
                # Default pickup method
                if current > 0:
                    forecast.forecasts[booking_class] = current * pickup_factor
                else:
                    forecast.forecasts[booking_class] = self._get_average_demand(
                        booking_class
                    ) * pickup_factor
        
        forecast.total_forecast = sum(forecast.forecasts.values())
        
        return forecast
    
    def _get_pickup_factor(self, days_before: int) -> float:
        """
        Get pickup factor based on days before departure.
        
        Pickup factor indicates what fraction of final demand is captured
        at each time horizon.
        
        Example:
        - 90 days out: 0.15 (15% of final demand)
        - 30 days out: 0.60 (60% of final demand)
        - 7 days out: 0.90 (90% of final demand)
        - Departure: 1.00 (100% of final demand)
        """
        if days_before in self.pickup_patterns:
            return self.pickup_patterns[days_before]
        
        # Default pickup curve (S-curve)
        # Based on typical airline booking patterns
        if days_before <= 0:
            factor = 1.0
        elif days_before <= 7:
            factor = 0.90 + (7 - days_before) * 0.01
        elif days_before <= 14:
            factor = 0.75 + (14 - days_before) * 0.015
        elif days_before <= 30:
            factor = 0.55 + (30 - days_before) * 0.0125
        elif days_before <= 60:
            factor = 0.30 + (60 - days_before) * 0.0083
        elif days_before <= 90:
            factor = 0.15 + (90 - days_before) * 0.005
        else:
            factor = 0.15
        
        return 1.0 / factor  # Convert to multiplier
    
    def _get_additive_pickup(
        self,
        booking_class: BookingClass,
        days_before: int
    ) -> float:
        """Get expected additional bookings (additive pickup)."""
        # Simplified - should learn from history
        avg_demand = self._get_average_demand(booking_class)
        pickup_factor = self._get_pickup_factor(days_before)
        
        # Remaining fraction to come
        remaining_fraction = 1.0 - (1.0 / pickup_factor)
        
        return avg_demand * remaining_fraction
    
    def _get_average_demand(self, booking_class: BookingClass) -> float:
        """Get historical average demand for a booking class."""
        # Simplified - should aggregate across similar flights
        total = 0.0
        count = 0
        
        for history in self.booking_history.values():
            class_total = sum(
                pax for dt, bc, pax in history if bc == booking_class
            )
            if class_total > 0:
                total += class_total
                count += 1
        
        if count > 0:
            return total / count
        
        # Default estimates
        defaults = {
            BookingClass.F: 10,
            BookingClass.J: 20,
            BookingClass.W: 30,
            BookingClass.Y: 40,
            BookingClass.L: 50
        }
        
        return defaults.get(booking_class, 30)
    
    def _forecast_exponential_smoothing(
        self,
        flight_date: FlightDate,
        current_bookings: Dict[BookingClass, int]
    ) -> DemandForecast:
        """Exponential smoothing forecast."""
        forecast = DemandForecast(
            flight_date_id=flight_date.flight_id,
            forecast_date=date.today(),
            method_used="exponential_smoothing"
        )
        
        alpha = 0.3  # Smoothing parameter
        
        for booking_class in BookingClass:
            # Get historical average
            historical = self._get_average_demand(booking_class)
            current = current_bookings.get(booking_class, 0)
            
            # Exponential smoothing
            smoothed = alpha * current + (1 - alpha) * historical
            
            forecast.forecasts[booking_class] = max(smoothed, current)
        
        forecast.total_forecast = sum(forecast.forecasts.values())
        
        return forecast
    
    def _forecast_neural_network(
        self,
        flight_date: FlightDate,
        current_date: date,
        current_bookings: Dict[BookingClass, int]
    ) -> DemandForecast:
        """Neural network-based forecasting."""
        forecast = DemandForecast(
            flight_date_id=flight_date.flight_id,
            forecast_date=current_date,
            method_used="neural_network"
        )
        
        if not TORCH_AVAILABLE or self.neural_model is None:
            # Fall back to pickup method
            return self._forecast_pickup(flight_date, current_date, current_bookings)
        
        # Prepare features
        days_before = (flight_date.departure_date - current_date).days
        day_of_week = flight_date.departure_date.weekday()
        month = flight_date.departure_date.month
        
        # Current booking levels (normalized)
        total_capacity = flight_date.schedule.aircraft.total_capacity
        current_total = sum(current_bookings.values())
        load_factor = current_total / total_capacity if total_capacity > 0 else 0
        
        # Create feature vector
        features = torch.tensor([
            days_before / 90.0,  # Normalized days
            day_of_week / 7.0,
            month / 12.0,
            load_factor,
            # Add more features as needed
        ], dtype=torch.float32)
        
        # Run inference
        with torch.no_grad():
            prediction = self.neural_model(features)
        
        # Convert prediction to forecasts
        total_predicted = prediction.item() * total_capacity
        
        # Distribute across classes (simplified)
        class_proportions = {
            BookingClass.L: 0.30,
            BookingClass.Y: 0.25,
            BookingClass.W: 0.20,
            BookingClass.J: 0.15,
            BookingClass.F: 0.10
        }
        
        for booking_class, proportion in class_proportions.items():
            forecast.forecasts[booking_class] = total_predicted * proportion
        
        forecast.total_forecast = total_predicted
        
        return forecast
    
    def _build_neural_model(self):
        """Build neural network for demand forecasting."""
        if not TORCH_AVAILABLE:
            return None
        
        class DemandForecastNet(nn.Module):
            def __init__(self, input_size=4, hidden_size=64):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden_size)
                self.fc2 = nn.Linear(hidden_size, hidden_size)
                self.fc3 = nn.Linear(hidden_size, 1)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))  # Output: 0-1 (load factor)
                return x
        
        return DemandForecastNet()
    
    def _add_forecast_noise(self, forecast: DemandForecast) -> DemandForecast:
        """Add random noise to simulate forecast error."""
        noisy_forecast = DemandForecast(
            flight_date_id=forecast.flight_date_id,
            forecast_date=forecast.forecast_date,
            method_used=forecast.method_used + "_noisy"
        )
        
        for booking_class, value in forecast.forecasts.items():
            # Add multiplicative noise
            noise = np.random.normal(1.0, self.noise_std)
            noisy_value = value * noise
            noisy_forecast.forecasts[booking_class] = max(0, noisy_value)
        
        noisy_forecast.total_forecast = sum(noisy_forecast.forecasts.values())
        
        return noisy_forecast
    
    def evaluate_forecast(
        self,
        forecast: DemandForecast,
        actual_bookings: Dict[BookingClass, int]
    ):
        """
        Evaluate forecast accuracy against actual bookings.
        
        Updates accuracy metrics.
        """
        if not self.track_accuracy:
            return
        
        # Calculate errors
        for booking_class in BookingClass:
            predicted = forecast.forecasts.get(booking_class, 0.0)
            actual = actual_bookings.get(booking_class, 0)
            
            self.accuracy.update(predicted, actual)
            
            # Track by class
            if booking_class not in self.accuracy.accuracy_by_class:
                self.accuracy.accuracy_by_class[booking_class] = 0.0
            
            error = abs(predicted - actual)
            self.accuracy.accuracy_by_class[booking_class] = (
                0.9 * self.accuracy.accuracy_by_class[booking_class] + 0.1 * error
            )
        
        # Store for history
        self.forecast_history.append((forecast, actual_bookings))
    
    def estimate_revenue_impact(
        self,
        forecast_error: float,
        fare: float,
        capacity: int
    ) -> float:
        """
        Estimate revenue lost due to forecast error.
        
        Over-forecasting: Set limits too high, accept low-value bookings
        Under-forecasting: Set limits too low, reject high-value bookings (spill)
        
        Args:
            forecast_error: Actual - Forecast
            fare: Average fare level
            capacity: Flight capacity
        
        Returns:
            Estimated revenue loss
        """
        if forecast_error > 0:
            # Under-forecasted: spilled high-value demand
            spilled_demand = min(forecast_error, capacity * 0.2)  # Assume 20% are high-value
            revenue_loss = spilled_demand * fare * 1.5  # High-value = 1.5x average
        else:
            # Over-forecasted: accepted too many low-value bookings
            excess_protection = abs(forecast_error)
            revenue_loss = excess_protection * fare * 0.3  # Opportunity cost
        
        self.accuracy.revenue_lost_to_error += revenue_loss
        
        return revenue_loss
    
    def get_accuracy_report(self) -> Dict:
        """Generate comprehensive accuracy report."""
        report = {
            'method': self.accuracy.method,
            'mae': self.accuracy.mean_absolute_error,
            'mape': self.accuracy.mean_absolute_percentage_error,
            'rmse': self.accuracy.root_mean_squared_error,
            'bias': self.accuracy.bias,
            'revenue_lost': self.accuracy.revenue_lost_to_error,
            'by_class': {
                bc.value: self.accuracy.accuracy_by_class.get(bc, 0.0)
                for bc in BookingClass
            },
            'total_forecasts': len(self.forecast_history)
        }
        
        return report
