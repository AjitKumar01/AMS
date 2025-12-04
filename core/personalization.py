"""
Personalization Engine for Airline Revenue Management.

Based on research by Wittman & Belobaba (2017) and Wang et al. (2021),
this module implements dynamic offer generation and personalized pricing.

Key Concepts:
1. Ancillary Bundling: Creating bundles based on customer preferences.
2. Dynamic Pricing: Adjusting fares based on loyalty status and estimated WTP.
3. Assortment Optimization: Showing the most relevant products to the user.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum
import random
import numpy as np

from core.models import BookingClass, Customer, CustomerSegment, TravelSolution
import copy

class LoyaltyTier(Enum):
    NONE = "none"
    SILVER = "silver"
    GOLD = "gold"
    PLATINUM = "platinum"

class AncillaryProduct(Enum):
    CHECKED_BAG = "checked_bag"
    PRIORITY_BOARDING = "priority_boarding"
    LOUNGE_ACCESS = "lounge_access"
    WIFI = "wifi"
    EXTRA_LEGROOM = "extra_legroom"

class PersonalizationEngine:
    """
    Engine to generate personalized offers for customers.
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        
        # Ancillary prices (base)
        self.ancillary_prices = {
            AncillaryProduct.CHECKED_BAG: 30.0,
            AncillaryProduct.PRIORITY_BOARDING: 20.0,
            AncillaryProduct.LOUNGE_ACCESS: 50.0,
            AncillaryProduct.WIFI: 15.0,
            AncillaryProduct.EXTRA_LEGROOM: 40.0
        }
    
    def enrich_customer(self, customer: Customer) -> Customer:
        """
        Add personalization attributes to a customer.
        Simulates retrieving data from a CRM.
        """
        if not self.enabled:
            return customer
            
        # Assign Loyalty Tier based on segment (probabilistic)
        # Business travelers are more likely to have status
        if customer.segment == CustomerSegment.BUSINESS:
            probs = [0.4, 0.3, 0.2, 0.1] # None, Silver, Gold, Platinum
        elif customer.segment == CustomerSegment.PREMIUM_LEISURE:
            probs = [0.6, 0.3, 0.1, 0.0]
        else:
            probs = [0.9, 0.08, 0.02, 0.0]
            
        tier_idx = np.random.choice(len(LoyaltyTier), p=probs)
        customer.loyalty_tier = list(LoyaltyTier)[tier_idx]
        
        # Assign Ancillary Preferences
        customer.ancillary_preferences = []
        if customer.segment in [CustomerSegment.LEISURE, CustomerSegment.VFR]:
            if random.random() < 0.7: customer.ancillary_preferences.append(AncillaryProduct.CHECKED_BAG.value)
        
        if customer.segment == CustomerSegment.BUSINESS:
            if random.random() < 0.6: customer.ancillary_preferences.append(AncillaryProduct.WIFI.value)
            if random.random() < 0.4: customer.ancillary_preferences.append(AncillaryProduct.LOUNGE_ACCESS.value)
            
        return customer

    def personalize_solutions(
        self, 
        solutions: List[TravelSolution],
        customer: Customer
    ) -> List[TravelSolution]:
        """
        Apply personalization to a list of travel solutions.
        Returns a new list of solutions including personalized offers.
        """
        if not self.enabled:
            return solutions
            
        personalized_solutions = []
        
        for sol in solutions:
            # Keep the original solution (Standard Offer)
            personalized_solutions.append(sol)
            
            # Check for Loyalty Discount
            if hasattr(customer, 'loyalty_tier') and customer.loyalty_tier in [LoyaltyTier.GOLD, LoyaltyTier.PLATINUM]:
                discount_sol = copy.deepcopy(sol)
                discount_sol.solution_id = f"{sol.solution_id}_loyalty"
                discount_sol.original_price = sol.total_price
                
                discount = 0.05 if customer.loyalty_tier == LoyaltyTier.GOLD else 0.10
                discount_sol.total_price *= (1.0 - discount)
                discount_sol.is_personalized_offer = True
                personalized_solutions.append(discount_sol)
            
            # Check for Bundling (Upsell)
            # If customer wants ancillaries, create a bundle
            if hasattr(customer, 'ancillary_preferences') and customer.ancillary_preferences:
                bundle_sol = copy.deepcopy(sol)
                bundle_sol.solution_id = f"{sol.solution_id}_bundle"
                bundle_sol.original_price = sol.total_price
                
                # Calculate bundle cost
                ancillary_cost = 0.0
                bundle_items = []
                for pref_str in customer.ancillary_preferences:
                    # Find enum from string value
                    try:
                        item = next(a for a in AncillaryProduct if a.value == pref_str)
                        cost = self.ancillary_prices.get(item, 0.0)
                        ancillary_cost += cost
                        bundle_items.append(item.value)
                    except StopIteration:
                        continue
                
                # Apply bundle discount (20% off ancillaries)
                bundle_price = sol.total_price + (ancillary_cost * 0.8)
                
                bundle_sol.total_price = bundle_price
                bundle_sol.ancillaries = bundle_items
                bundle_sol.is_personalized_offer = True
                
                personalized_solutions.append(bundle_sol)
                
        return personalized_solutions

