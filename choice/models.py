"""
Customer choice models for airline revenue management.

Implements utility-based choice modeling including:
- Multinomial Logit (MNL) - industry standard
- Buy-up/buy-down behavior
- Recapture modeling
- Competitor consideration
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import numpy as np
from abc import ABC, abstractmethod

from core.models import (
    Customer, TravelSolution, BookingClass, 
    CustomerSegment, Airport
)


class ChoiceMethod(Enum):
    """Customer choice modeling methods."""
    CHEAPEST = "cheapest"                    # Simple: pick cheapest
    MULTINOMIAL_LOGIT = "mnl"               # Utility-based MNL
    NESTED_LOGIT = "nested_logit"           # Two-level choice
    MIXED_LOGIT = "mixed_logit"             # Random coefficients


@dataclass
class UtilityFunction:
    """
    Utility function for customer choice modeling.
    
    U = β_price * price + β_time * time + β_direct * is_direct + 
        β_schedule * schedule_quality + β_loyalty * loyalty_bonus + ε
    
    Where ε ~ Gumbel(0, μ) for MNL
    """
    
    # Coefficients (typically negative for costs)
    price_coef: float = -0.002              # Per dollar
    time_coef: float = -0.01                # Per minute
    connection_penalty: float = -0.5        # Per connection
    direct_flight_bonus: float = 1.0        # Direct vs connecting
    
    # Schedule convenience (departure time preference)
    early_morning_penalty: float = -0.3     # Before 6 AM
    late_night_penalty: float = -0.4        # After 10 PM
    
    # Loyalty bonus
    loyalty_coef: float = 0.5               # Loyalty program benefit
    
    # Segment-specific adjustments
    business_time_multiplier: float = 2.0   # Business values time more
    business_price_multiplier: float = 0.5  # Business less price-sensitive
    
    # Scale parameter for logit (inverse of variance of error term)
    scale_parameter: float = 1.0
    
    # Non-linear price utility
    use_log_price: bool = False
    
    def calculate_deterministic_utility(
        self,
        solution: TravelSolution,
        customer: Customer,
        has_loyalty: bool = False
    ) -> float:
        """
        Calculate deterministic part of utility.
        
        Args:
            solution: Travel solution to evaluate
            customer: Customer making choice
            has_loyalty: Whether customer has loyalty status
            
        Returns:
            Deterministic utility value
        """
        utility = 0.0
        
        # Price disutility
        price_coef = self.price_coef
        if customer.segment == CustomerSegment.BUSINESS:
            price_coef *= self.business_price_multiplier
            
        if self.use_log_price and solution.total_price > 0:
            # Log-linear price utility: U = beta * ln(price)
            # This implies diminishing sensitivity at higher price points
            utility += price_coef * np.log(solution.total_price)
        else:
            # Linear price utility: U = beta * price
            utility += price_coef * solution.total_price
        
        # Travel time disutility
        time_coef = self.time_coef
        if customer.segment == CustomerSegment.BUSINESS:
            time_coef *= self.business_time_multiplier
        
        travel_minutes = solution.total_travel_time.total_seconds() / 60
        utility += time_coef * travel_minutes
        
        # Connection penalties
        if solution.is_direct:
            utility += self.direct_flight_bonus
        else:
            utility += self.connection_penalty * solution.num_connections
        
        # Schedule convenience
        departure_hour = solution.departure_time.hour
        if departure_hour < 6:
            utility += self.early_morning_penalty
        elif departure_hour >= 22:
            utility += self.late_night_penalty
        
        # Loyalty bonus
        if has_loyalty:
            utility += self.loyalty_coef * customer.loyalty_score
        
        return utility
    
    def calculate_total_utility(
        self,
        solution: TravelSolution,
        customer: Customer,
        has_loyalty: bool = False,
        rng: Optional[np.random.Generator] = None
    ) -> float:
        """
        Calculate total utility including random component.
        
        Args:
            solution: Travel solution to evaluate
            customer: Customer making choice
            has_loyalty: Whether customer has loyalty status
            rng: Random number generator for error term
            
        Returns:
            Total utility (deterministic + random)
        """
        det_utility = self.calculate_deterministic_utility(
            solution, customer, has_loyalty
        )
        
        # Add Gumbel-distributed error term for MNL
        if rng is not None:
            # Gumbel(0, scale) using inverse transform
            u = rng.uniform(0, 1)
            epsilon = -self.scale_parameter * np.log(-np.log(u))
            return det_utility + epsilon
        
        return det_utility


@dataclass
class ChoiceSet:
    """
    Set of alternatives available to customer.
    
    Includes own airline and competitor options.
    """
    own_solutions: List[TravelSolution] = field(default_factory=list)
    competitor_solutions: List[TravelSolution] = field(default_factory=list)
    no_purchase_utility: float = 0.0  # Utility of not buying (outside option)
    
    @property
    def all_solutions(self) -> List[TravelSolution]:
        """All purchase options."""
        return self.own_solutions + self.competitor_solutions
    
    @property
    def has_available_options(self) -> bool:
        """Check if any solutions are available."""
        return any(s.available_seats > 0 for s in self.all_solutions)


class ChoiceModel(ABC):
    """Abstract base class for choice models."""
    
    @abstractmethod
    def predict_choice(
        self,
        choice_set: ChoiceSet,
        customer: Customer,
        rng: np.random.Generator
    ) -> Optional[TravelSolution]:
        """
        Predict customer's choice from available alternatives.
        
        Returns:
            Selected solution or None (no purchase)
        """
        pass
    
    @abstractmethod
    def calculate_choice_probabilities(
        self,
        choice_set: ChoiceSet,
        customer: Customer
    ) -> Dict[int, float]:
        """
        Calculate probability of choosing each alternative.
        
        Returns:
            Dict mapping solution index to choice probability
        """
        pass


class MultinomialLogitModel(ChoiceModel):
    """
    Multinomial Logit (MNL) choice model.
    
    Standard model in airline revenue management:
    P(i) = exp(U_i) / Σ_j exp(U_j)
    
    Where U_i is the utility of alternative i.
    """
    
    def __init__(
        self,
        utility_function: Optional[UtilityFunction] = None,
        include_no_purchase: bool = True
    ):
        """
        Initialize MNL model.
        
        Args:
            utility_function: Function for computing utilities
            include_no_purchase: Whether to include no-purchase option
        """
        self.utility_fn = utility_function or UtilityFunction()
        self.include_no_purchase = include_no_purchase
    
    def calculate_choice_probabilities(
        self,
        choice_set: ChoiceSet,
        customer: Customer
    ) -> Dict[int, float]:
        """
        Calculate choice probabilities using MNL formula.
        
        Args:
            choice_set: Available alternatives
            customer: Customer making choice
            
        Returns:
            Dict mapping solution index to probability
        """
        # Filter to available solutions
        available = [s for s in choice_set.all_solutions if s.available_seats > 0]
        
        if not available:
            return {-1: 1.0}  # -1 represents no purchase
        
        # Calculate deterministic utilities
        utilities = []
        for solution in available:
            has_loyalty = customer.loyalty_score > 0.5
            utility = self.utility_fn.calculate_deterministic_utility(
                solution, customer, has_loyalty
            )
            utilities.append(utility)
        
        # Add no-purchase option
        if self.include_no_purchase:
            utilities.append(choice_set.no_purchase_utility)
        
        # Convert to numpy array
        utilities_array = np.array(utilities)
        
        # Numerical stability: subtract max
        utilities_array = utilities_array - np.max(utilities_array)
        
        # Calculate probabilities
        exp_utilities = np.exp(utilities_array)
        probabilities = exp_utilities / np.sum(exp_utilities)
        
        # Map to solution indices
        result = {}
        for i, prob in enumerate(probabilities):
            if i < len(available):
                result[i] = prob
            else:
                result[-1] = prob  # No purchase
        
        return result
    
    def predict_choice(
        self,
        choice_set: ChoiceSet,
        customer: Customer,
        rng: np.random.Generator
    ) -> Optional[TravelSolution]:
        """
        Predict choice using MNL with random utility realization.
        
        Args:
            choice_set: Available alternatives
            customer: Customer making choice
            rng: Random number generator
            
        Returns:
            Chosen solution or None
        """
        # Filter to available solutions
        available = [s for s in choice_set.all_solutions if s.available_seats > 0]
        
        if not available:
            return None
        
        # Calculate total utilities (including random component)
        utilities = []
        for solution in available:
            has_loyalty = customer.loyalty_score > 0.5
            utility = self.utility_fn.calculate_total_utility(
                solution, customer, has_loyalty, rng
            )
            utilities.append(utility)
        
        # Add no-purchase option
        if self.include_no_purchase:
            # No-purchase utility with random component
            u = rng.uniform(0, 1)
            epsilon = -self.utility_fn.scale_parameter * np.log(-np.log(u))
            no_purchase_total_utility = choice_set.no_purchase_utility + epsilon
            utilities.append(no_purchase_total_utility)
        
        # Choose alternative with highest total utility
        max_idx = np.argmax(utilities)
        
        if max_idx < len(available):
            return available[max_idx]
        else:
            return None  # No purchase chosen


@dataclass
class BuyUpDownModel:
    """
    Model for buy-up and buy-down behavior.
    
    When preferred class is unavailable, customers may:
    - Buy up to a higher class (with some probability)
    - Buy down to a lower class (with some probability)
    - Not purchase at all
    """
    
    # Buy-up probabilities by segment
    buy_up_probs: Dict[CustomerSegment, float] = field(default_factory=lambda: {
        CustomerSegment.BUSINESS: 0.40,  # Business willing to buy up
        CustomerSegment.LEISURE: 0.10,   # Leisure rarely buys up
        CustomerSegment.VFR: 0.05,
        CustomerSegment.GROUP: 0.02
    })
    
    # Buy-down probabilities by segment
    buy_down_probs: Dict[CustomerSegment, float] = field(default_factory=lambda: {
        CustomerSegment.BUSINESS: 0.15,  # Business rarely buys down
        CustomerSegment.LEISURE: 0.50,   # Leisure often buys down
        CustomerSegment.VFR: 0.45,
        CustomerSegment.GROUP: 0.40
    })
    
    # Maximum price increase willing to accept for buy-up (as ratio)
    max_buyup_price_ratio: Dict[CustomerSegment, float] = field(default_factory=lambda: {
        CustomerSegment.BUSINESS: 1.5,   # Up to 50% more
        CustomerSegment.LEISURE: 1.2,    # Up to 20% more
        CustomerSegment.VFR: 1.15,
        CustomerSegment.GROUP: 1.10
    })
    
    def will_buy_up(
        self,
        customer: Customer,
        preferred_price: float,
        higher_price: float,
        rng: np.random.Generator
    ) -> bool:
        """
        Determine if customer will buy up to higher class.
        
        Args:
            customer: Customer making decision
            preferred_price: Price of preferred class (unavailable)
            higher_price: Price of higher class
            rng: Random number generator
            
        Returns:
            True if customer buys up
        """
        # Check price tolerance
        price_ratio = higher_price / preferred_price if preferred_price > 0 else 999
        max_ratio = self.max_buyup_price_ratio.get(customer.segment, 1.3)
        
        if price_ratio > max_ratio:
            return False
        
        # Check against willingness to pay
        if higher_price > customer.willingness_to_pay:
            return False
        
        # Stochastic decision
        buy_up_prob = self.buy_up_probs.get(customer.segment, 0.15)
        
        # Adjust probability based on price increase
        prob_adjustment = 1.0 - (price_ratio - 1.0) / (max_ratio - 1.0)
        adjusted_prob = buy_up_prob * prob_adjustment
        
        return rng.random() < adjusted_prob
    
    def will_buy_down(
        self,
        customer: Customer,
        preferred_price: float,
        lower_price: float,
        rng: np.random.Generator
    ) -> bool:
        """
        Determine if customer will buy down to lower class.
        
        Args:
            customer: Customer making decision
            preferred_price: Price of preferred class (unavailable)
            lower_price: Price of lower class
            rng: Random number generator
            
        Returns:
            True if customer buys down
        """
        # Always prefer cheaper if it meets needs
        buy_down_prob = self.buy_down_probs.get(customer.segment, 0.30)
        
        # Higher probability if significant savings
        savings_ratio = (preferred_price - lower_price) / preferred_price
        prob_adjustment = 1.0 + savings_ratio  # More savings = higher probability
        adjusted_prob = min(0.9, buy_down_prob * prob_adjustment)
        
        return rng.random() < adjusted_prob


@dataclass
class RecaptureModel:
    """
    Model for recapture rates when preferred product unavailable.
    
    Recapture = probability customer accepts alternative instead of
    going to competitor or not purchasing.
    """
    
    # Base recapture rates by segment
    base_recapture_rates: Dict[CustomerSegment, float] = field(default_factory=lambda: {
        CustomerSegment.BUSINESS: 0.60,  # Business has higher recapture
        CustomerSegment.LEISURE: 0.40,   # Leisure more likely to shop
        CustomerSegment.VFR: 0.50,
        CustomerSegment.GROUP: 0.70      # Groups have less flexibility
    })
    
    # Recapture rate adjustments
    loyalty_bonus: float = 0.20          # Loyal customers more likely to accept
    price_sensitivity_penalty: float = -0.15  # Price-sensitive more likely to leave
    
    def get_recapture_probability(
        self,
        customer: Customer,
        has_loyalty: bool = False
    ) -> float:
        """
        Calculate probability of recapturing customer to alternative.
        
        Args:
            customer: Customer to evaluate
            has_loyalty: Whether customer has loyalty status
            
        Returns:
            Recapture probability (0-1)
        """
        base_rate = self.base_recapture_rates.get(customer.segment, 0.45)
        
        # Loyalty adjustment
        if has_loyalty and customer.loyalty_score > 0.5:
            base_rate += self.loyalty_bonus
        
        # Price sensitivity adjustment
        if customer.price_sensitivity > 1.3:
            base_rate += self.price_sensitivity_penalty
        
        # Cap between 0 and 1
        return max(0.0, min(1.0, base_rate))
    
    def will_accept_alternative(
        self,
        customer: Customer,
        alternative_utility: float,
        competitor_utility: float,
        has_loyalty: bool,
        rng: np.random.Generator
    ) -> bool:
        """
        Determine if customer accepts alternative or goes to competitor.
        
        Args:
            customer: Customer making decision
            alternative_utility: Utility of offered alternative
            competitor_utility: Estimated utility of competitor option
            has_loyalty: Whether customer has loyalty status
            rng: Random number generator
            
        Returns:
            True if customer accepts alternative
        """
        recapture_prob = self.get_recapture_probability(customer, has_loyalty)
        
        # Adjust based on relative utilities
        if alternative_utility > competitor_utility:
            # Alternative is better - higher recapture
            utility_boost = 0.2
            recapture_prob = min(1.0, recapture_prob + utility_boost)
        elif alternative_utility < competitor_utility * 0.8:
            # Alternative much worse - lower recapture
            utility_penalty = 0.2
            recapture_prob = max(0.0, recapture_prob - utility_penalty)
        
        return rng.random() < recapture_prob


class EnhancedChoiceModel:
    """
    Enhanced choice model combining MNL with buy-up/down and recapture.
    
    This is the recommended model for realistic revenue management simulation.
    """
    
    def __init__(
        self,
        mnl_model: Optional[MultinomialLogitModel] = None,
        buyupdown_model: Optional[BuyUpDownModel] = None,
        recapture_model: Optional[RecaptureModel] = None
    ):
        """
        Initialize enhanced choice model.
        
        Args:
            mnl_model: Base MNL model
            buyupdown_model: Buy-up/down behavior model
            recapture_model: Recapture model
        """
        self.mnl = mnl_model or MultinomialLogitModel()
        self.buyupdown = buyupdown_model or BuyUpDownModel()
        self.recapture = recapture_model or RecaptureModel()
    
    def predict_choice_with_behavior(
        self,
        choice_set: ChoiceSet,
        customer: Customer,
        preferred_class: Optional[BookingClass],
        rng: np.random.Generator
    ) -> Optional[TravelSolution]:
        """
        Predict choice considering buy-up/down and recapture behavior.
        
        Args:
            choice_set: Available alternatives
            customer: Customer making choice
            preferred_class: Customer's preferred booking class (if known)
            rng: Random number generator
            
        Returns:
            Chosen solution or None
        """
        # First, try standard MNL choice
        chosen = self.mnl.predict_choice(choice_set, customer, rng)
        
        if chosen is not None:
            return chosen
        
        # If no choice made, customer might reconsider with buy-up/down
        available = [s for s in choice_set.all_solutions if s.available_seats > 0]
        
        if not available:
            return None
        
        # Check for buy-up/down opportunities
        if preferred_class and len(available) > 1:
            # Sort by price
            sorted_solutions = sorted(available, key=lambda s: s.total_price)
            
            # Try buy-down first (cheaper option)
            for solution in sorted_solutions:
                if solution.total_price < customer.willingness_to_pay * 0.8:
                    if self.buyupdown.will_buy_down(
                        customer, 
                        customer.willingness_to_pay,
                        solution.total_price,
                        rng
                    ):
                        return solution
            
            # Try buy-up (more expensive option)
            for solution in reversed(sorted_solutions):
                if solution.total_price <= customer.willingness_to_pay:
                    if self.buyupdown.will_buy_up(
                        customer,
                        customer.willingness_to_pay * 0.7,
                        solution.total_price,
                        rng
                    ):
                        return solution
        
        # Check recapture
        if available:
            best_solution = min(available, key=lambda s: s.total_price)
            has_loyalty = customer.loyalty_score > 0.5
            
            # Estimate utilities
            alt_utility = self.mnl.utility_fn.calculate_deterministic_utility(
                best_solution, customer, has_loyalty
            )
            competitor_utility = 0.0  # Simplified
            
            if self.recapture.will_accept_alternative(
                customer, alt_utility, competitor_utility, has_loyalty, rng
            ):
                return best_solution
        
        return None
