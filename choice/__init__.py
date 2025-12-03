"""
Customer choice modeling for airline revenue management.

Handles:
- Multinomial Logit (MNL) choice model
- Buy-up and buy-down behavior
- Recapture rates when preferred choice unavailable
- Competitor switching probabilities
- Utility-based choice modeling
"""

from .models import (
    ChoiceModel,
    MultinomialLogitModel,
    UtilityFunction,
    BuyUpDownModel,
    RecaptureModel,
    ChoiceSet
)

__all__ = [
    'ChoiceModel',
    'MultinomialLogitModel',
    'UtilityFunction',
    'BuyUpDownModel',
    'RecaptureModel',
    'ChoiceSet'
]
