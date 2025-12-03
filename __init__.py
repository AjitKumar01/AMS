"""PyAirline RM - Advanced Airline Revenue Management Simulator."""

__version__ = "0.1.0"
__author__ = "Your Name"
__license__ = "MIT"

from core.simulator import Simulator, SimulationConfig, SimulationResults
from core.models import *
from core.events import EventManager

__all__ = [
    'Simulator',
    'SimulationConfig',
    'SimulationResults',
    'EventManager',
]
