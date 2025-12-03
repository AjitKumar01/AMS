"""
Inventory management module.

Handles:
- Availability checking
- Booking limit management
- Nested inventory control
- Overbooking optimization
"""

from .network import (
    NetworkOptimizer,
    VirtualBucket,
    DisplacementCost
)

__all__ = [
    'NetworkOptimizer',
    'VirtualBucket',
    'DisplacementCost'
]
