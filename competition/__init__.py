"""
Multi-airline competition module for realistic market simulation.

This module provides:
- Airline agents with different competitive strategies
- Market share tracking and dynamics
- Competitive pricing behaviors
- Capacity responses
"""

from .airline import Airline, CompetitiveStrategy
from .market import Market, MarketSegment
from .strategies import (
    AggressiveStrategy,
    ConservativeStrategy,
    MLBasedStrategy,
    MatchCompetitorStrategy
)

__all__ = [
    'Airline',
    'CompetitiveStrategy',
    'Market',
    'MarketSegment',
    'AggressiveStrategy',
    'ConservativeStrategy',
    'MLBasedStrategy',
    'MatchCompetitorStrategy'
]
