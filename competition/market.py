"""
Market structure and dynamics for multi-airline competition.

Tracks:
- Market shares by airline
- Competitive dynamics
- Market segment performance
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import date
import numpy as np

from core.models import Route, Airport
from .airline import Airline


@dataclass
class MarketSegment:
    """A market segment (O-D pair) with multiple competing airlines."""
    origin: Airport
    destination: Airport
    total_demand: float = 0.0
    captured_demand: Dict[str, float] = field(default_factory=dict)  # airline_code -> demand
    total_revenue: Dict[str, float] = field(default_factory=dict)  # airline_code -> revenue
    
    def get_market_share(self, airline_code: str) -> float:
        """Get airline's market share by passengers."""
        total = sum(self.captured_demand.values())
        if total == 0:
            return 0.0
        return self.captured_demand.get(airline_code, 0.0) / total
    
    def get_revenue_share(self, airline_code: str) -> float:
        """Get airline's market share by revenue."""
        total = sum(self.total_revenue.values())
        if total == 0:
            return 0.0
        return self.total_revenue.get(airline_code, 0.0) / total
    
    @property
    def market_key(self) -> str:
        """Return market identifier."""
        return f"{self.origin.code}-{self.destination.code}"


class Market:
    """
    Market coordinator for competitive simulation.
    
    Manages:
    - Market segments (O-D pairs)
    - Information flow between airlines
    - Market equilibrium
    """
    
    def __init__(self, information_transparency: float = 0.7):
        """
        Initialize market.
        
        Args:
            information_transparency: How much airlines can observe (0-1)
                1.0 = perfect information
                0.0 = no information about competitors
        """
        self.segments: Dict[str, MarketSegment] = {}
        self.airlines: Dict[str, Airline] = {}
        self.information_transparency = information_transparency
        
        # Track market history
        self.history: List[Dict] = []
    
    def add_airline(self, airline: Airline):
        """Add airline to the market."""
        self.airlines[airline.code] = airline
    
    def get_or_create_segment(self, origin: Airport, destination: Airport) -> MarketSegment:
        """Get existing segment or create new one."""
        key = f"{origin.code}-{destination.code}"
        if key not in self.segments:
            self.segments[key] = MarketSegment(origin=origin, destination=destination)
        return self.segments[key]
    
    def record_booking(
        self,
        airline_code: str,
        origin: Airport,
        destination: Airport,
        passengers: int,
        revenue: float
    ):
        """Record a booking in the market."""
        segment = self.get_or_create_segment(origin, destination)
        
        if airline_code not in segment.captured_demand:
            segment.captured_demand[airline_code] = 0.0
            segment.total_revenue[airline_code] = 0.0
        
        segment.captured_demand[airline_code] += passengers
        segment.total_revenue[airline_code] += revenue
        segment.total_demand += passengers
    
    def share_competitive_intelligence(self, current_date: date):
        """
        Share information between airlines based on transparency level.
        
        Airlines observe:
        - Published fares (always visible)
        - Load factors (partially visible)
        - Capacity (observable)
        """
        for segment_key, segment in self.segments.items():
            # For each airline in this market
            for observer_code, observer in self.airlines.items():
                # Observe competitors
                for competitor_code, competitor in self.airlines.items():
                    if observer_code == competitor_code:
                        continue
                    
                    # Share information based on transparency
                    if np.random.random() < self.information_transparency:
                        # Observe competitor fares
                        for fd in competitor.flight_dates.values():
                            fd_key = f"{fd.schedule.route.origin.code}-{fd.schedule.route.destination.code}"
                            if fd_key == segment_key:
                                # Get published fare
                                if fd.fares:
                                    avg_fare = np.mean([f.base_amount for f in fd.fares.values()])
                                    observer.observe_competitor_fare(
                                        competitor_code,
                                        segment_key,
                                        avg_fare,
                                        current_date
                                    )
                                
                                # Observe load factor (with noise)
                                lf = fd.load_factor()
                                noise = np.random.normal(0, 0.05)  # ±5% noise
                                observed_lf = np.clip(lf + noise, 0, 1)
                                observer.observe_competitor_load_factor(
                                    competitor_code,
                                    segment_key,
                                    observed_lf
                                )
    
    def calculate_herfindahl_index(self, segment_key: str) -> float:
        """
        Calculate Herfindahl-Hirschman Index (HHI) for market concentration.
        
        HHI = sum of squared market shares (0-10000)
        - < 1500: Competitive market
        - 1500-2500: Moderate concentration
        - > 2500: Highly concentrated
        """
        if segment_key not in self.segments:
            return 0.0
        
        segment = self.segments[segment_key]
        total = sum(segment.captured_demand.values())
        
        if total == 0:
            return 0.0
        
        hhi = 0.0
        for demand in segment.captured_demand.values():
            share = (demand / total) * 100  # Convert to percentage
            hhi += share ** 2
        
        return hhi
    
    def get_market_summary(self, segment_key: str) -> Dict:
        """Get summary statistics for a market segment."""
        if segment_key not in self.segments:
            return {}
        
        segment = self.segments[segment_key]
        
        summary = {
            'segment': segment_key,
            'total_demand': segment.total_demand,
            'total_revenue': sum(segment.total_revenue.values()),
            'num_competitors': len(segment.captured_demand),
            'hhi': self.calculate_herfindahl_index(segment_key),
            'airlines': {}
        }
        
        # Per-airline metrics
        for airline_code in segment.captured_demand.keys():
            summary['airlines'][airline_code] = {
                'passengers': segment.captured_demand[airline_code],
                'revenue': segment.total_revenue.get(airline_code, 0.0),
                'market_share_pax': segment.get_market_share(airline_code),
                'market_share_revenue': segment.get_revenue_share(airline_code),
                'average_fare': (
                    segment.total_revenue.get(airline_code, 0.0) / 
                    segment.captured_demand[airline_code]
                    if segment.captured_demand[airline_code] > 0 else 0.0
                )
            }
        
        return summary
    
    def record_snapshot(self, current_date: date):
        """Record market state for historical analysis."""
        snapshot = {
            'date': current_date,
            'segments': {}
        }
        
        for segment_key in self.segments.keys():
            snapshot['segments'][segment_key] = self.get_market_summary(segment_key)
        
        self.history.append(snapshot)
    
    def analyze_competitive_dynamics(self) -> Dict:
        """
        Analyze competitive dynamics across all markets.
        
        Returns:
            Analysis of competition intensity, pricing power, etc.
        """
        analysis = {
            'overall_hhi': 0.0,
            'avg_competitors_per_market': 0.0,
            'price_dispersion': {},
            'market_concentration': {}
        }
        
        if not self.segments:
            return analysis
        
        # Average HHI across markets
        hhis = [self.calculate_herfindahl_index(k) for k in self.segments.keys()]
        analysis['overall_hhi'] = np.mean(hhis) if hhis else 0.0
        
        # Average competitors
        competitors_count = [
            len(seg.captured_demand) for seg in self.segments.values()
        ]
        analysis['avg_competitors_per_market'] = (
            np.mean(competitors_count) if competitors_count else 0.0
        )
        
        # Classify market concentration
        for segment_key in self.segments.keys():
            hhi = self.calculate_herfindahl_index(segment_key)
            if hhi < 1500:
                concentration = "competitive"
            elif hhi < 2500:
                concentration = "moderate"
            else:
                concentration = "concentrated"
            
            analysis['market_concentration'][segment_key] = {
                'hhi': hhi,
                'classification': concentration
            }
        
        return analysis
