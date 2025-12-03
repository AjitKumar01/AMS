"""
Price sensitivity and demand elasticity models.

This module implements price elasticity of demand logic to support
price-sensitive revenue management. It allows adjusting demand forecasts
based on price changes relative to reference fares.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
from enum import Enum
import numpy as np

from core.models import CustomerSegment, BookingClass


@dataclass
class ElasticityModel:
    """
    Price elasticity model for a customer segment or booking class.
    
    Formula:
        Demand_New = Demand_Ref * (Price_New / Price_Ref) ^ Elasticity
        
    Elasticity is typically negative:
    - Elasticity < -1: Elastic (Price increase -> Revenue decrease)
    - Elasticity = -1: Unit elastic
    - -1 < Elasticity < 0: Inelastic (Price increase -> Revenue increase)
    """
    
    # Default elasticities by segment (typical airline values)
    segment_elasticities: Dict[CustomerSegment, float] = field(default_factory=lambda: {
        CustomerSegment.BUSINESS: -0.4,   # Inelastic (less sensitive)
        CustomerSegment.LEISURE: -1.8,    # Elastic (sensitive)
        CustomerSegment.VFR: -1.2,        # Moderately elastic
        CustomerSegment.GROUP: -2.0       # Highly elastic
    })
    
    # Default elasticities by booking class (if segment not available)
    class_elasticities: Dict[BookingClass, float] = field(default_factory=lambda: {
        BookingClass.F: -0.2,  # First class: very inelastic
        BookingClass.J: -0.4,  # Business: inelastic
        BookingClass.W: -0.8,  # Premium Eco: slightly inelastic
        BookingClass.Y: -1.0,  # Full Eco: unit elastic
        BookingClass.B: -1.2,
        BookingClass.M: -1.4,
        BookingClass.H: -1.6,
        BookingClass.Q: -1.8,
        BookingClass.K: -2.0,
        BookingClass.L: -2.5   # Deep discount: very elastic
    })
    
    def adjust_demand(
        self,
        base_demand: float,
        current_price: float,
        reference_price: float,
        segment: Optional[CustomerSegment] = None,
        booking_class: Optional[BookingClass] = None
    ) -> float:
        """
        Calculate adjusted demand based on price elasticity.
        
        Args:
            base_demand: Demand at reference price
            current_price: New price to test
            reference_price: Reference price (baseline)
            segment: Customer segment (preferred for elasticity lookup)
            booking_class: Booking class (fallback for elasticity lookup)
            
        Returns:
            Adjusted demand forecast
        """
        if base_demand <= 0 or current_price <= 0 or reference_price <= 0:
            return base_demand
            
        # Determine elasticity coefficient
        elasticity = -1.0  # Default
        
        if segment and segment in self.segment_elasticities:
            elasticity = self.segment_elasticities[segment]
        elif booking_class and booking_class in self.class_elasticities:
            elasticity = self.class_elasticities[booking_class]
            
        # Calculate adjustment factor
        # Limit the price ratio to avoid extreme values
        price_ratio = current_price / reference_price
        price_ratio = max(0.5, min(2.0, price_ratio))
        
        adjustment_factor = np.power(price_ratio, elasticity)
        
        return base_demand * adjustment_factor

    def get_optimal_price(
        self,
        base_demand: float,
        reference_price: float,
        capacity: int,
        segment: CustomerSegment
    ) -> float:
        """
        Calculate revenue-maximizing price given capacity constraint.
        
        Note: This is a simplified unconstrained optimization.
        """
        elasticity = self.segment_elasticities.get(segment, -1.5)
        
        # If inelastic (|e| < 1), raise price as much as possible (theoretically infinite)
        # In practice, bounded by max fare or competition.
        if abs(elasticity) < 1.0:
            return reference_price * 1.5
            
        # If elastic, we want to find price that fills capacity or maximizes revenue
        # Revenue R(p) = p * D(p) = p * D0 * (p/p0)^e = C * p^(1+e)
        # If e < -1, Revenue is maximized at lower prices (but limited by capacity)
        
        # Simple heuristic: Find price that generates demand = capacity
        # Capacity = D0 * (p/p0)^e
        # (Capacity/D0) = (p/p0)^e
        # (Capacity/D0)^(1/e) = p/p0
        # p = p0 * (Capacity/D0)^(1/e)
        
        if base_demand > capacity:
            # Demand exceeds capacity, raise price to spill excess
            optimal_price = reference_price * np.power(capacity / base_demand, 1.0 / elasticity)
            return optimal_price
        else:
            # Demand < capacity, lower price to stimulate demand (if elastic)
            # But don't go below marginal cost (not modeled here, assume min floor)
            optimal_price = reference_price * np.power(capacity / base_demand, 1.0 / elasticity)
            return max(optimal_price, reference_price * 0.5)
