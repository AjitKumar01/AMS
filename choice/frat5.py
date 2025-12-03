"""
FRAT5 (Fare Ratio at 50%) Sell-up Model.

This module implements the industry-standard FRAT5 curve for modeling 
sell-up (buy-up) behavior. This is used to estimate the probability 
that a customer will purchase a higher fare when their preferred 
lower fare is unavailable.

The standard formula is:
    P(buy_up) = 0.5 ^ ((Fare_High / Fare_Low - 1) / (FRAT5 - 1))

Where:
- Fare_High / Fare_Low is the Fare Ratio (r)
- FRAT5 is the ratio at which 50% of customers sell up
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np

from core.models import CustomerSegment

@dataclass
class FRAT5Model:
    """
    FRAT5 Sell-up Model.
    
    Attributes:
        segment_frat5: Dictionary mapping customer segments to their FRAT5 parameter.
                       Higher values mean LESS price sensitivity (more willing to buy up).
    """
    
    # Default industry values (approximate)
    # Business: Willing to pay 2.5x the base fare with 50% probability
    # Leisure: Willing to pay 1.5x the base fare with 50% probability
    segment_frat5: Dict[CustomerSegment, float] = field(default_factory=lambda: {
        CustomerSegment.BUSINESS: 2.5,
        CustomerSegment.LEISURE: 1.5,
        CustomerSegment.VFR: 1.3,
        CustomerSegment.GROUP: 1.2
    })
    
    def calculate_sellup_prob(
        self, 
        lower_fare: float, 
        higher_fare: float, 
        segment: CustomerSegment
    ) -> float:
        """
        Calculate the probability of selling up from lower_fare to higher_fare.
        
        Args:
            lower_fare: The unavailable lower fare (reference)
            higher_fare: The available higher fare
            segment: The customer segment
            
        Returns:
            Probability (0.0 to 1.0)
        """
        if lower_fare <= 0 or higher_fare <= 0:
            return 0.0
            
        # If higher fare is actually cheaper or same, assume 100% sell-up
        if higher_fare <= lower_fare:
            return 1.0
            
        fare_ratio = higher_fare / lower_fare
        frat5 = self.segment_frat5.get(segment, 1.5)
        
        # Safety check for FRAT5 <= 1.0 (would cause division by zero or invalid logic)
        if frat5 <= 1.001:
            return 0.0 if fare_ratio > 1.0 else 1.0
            
        # Standard FRAT5 Formula
        # P = 0.5 ^ ((r - 1) / (FRAT5 - 1))
        exponent = (fare_ratio - 1.0) / (frat5 - 1.0)
        probability = np.power(0.5, exponent)
        
        return float(probability)

    def get_max_acceptable_fare_ratio(self, segment: CustomerSegment, acceptance_prob: float = 0.1) -> float:
        """
        Calculate the fare ratio at which acceptance drops to a specific probability.
        Useful for determining the 'maximum' reasonable upsell.
        """
        frat5 = self.segment_frat5.get(segment, 1.5)
        if frat5 <= 1.0:
            return 1.0
            
        # Inverse formula
        # log0.5(P) = (r - 1) / (FRAT5 - 1)
        # r = 1 + (FRAT5 - 1) * log0.5(P)
        
        log_prob = np.log(acceptance_prob) / np.log(0.5)
        ratio = 1.0 + (frat5 - 1.0) * log_prob
        
        return ratio
