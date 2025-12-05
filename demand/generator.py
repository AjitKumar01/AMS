"""
Demand generation engine for realistic passenger booking requests.

This module generates booking requests with:
- Poisson arrival processes
- Log-normal WTP distributions
- Realistic booking curves (demand varies by DTD)
- Customer segmentation (business/leisure)
- Seasonality and special events
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, time
from typing import List, Dict, Optional, Tuple
import numpy as np
from scipy import stats
import logging

from core.models import (
    BookingRequest, Customer, CustomerSegment, Airport, 
    CabinClass, TripType, BookingChannel, MarketConditions
)
from core.events import EventManager, BookingRequestEvent, EventType


@dataclass
class DemandStreamConfig:
    """Configuration for a demand stream (O-D pair with specific characteristics)."""
    
    stream_id: str
    origin: Airport
    destination: Airport
    
    # Demand volume
    mean_daily_demand: float = 100.0  # Average bookings per day
    demand_std: float = 20.0  # Standard deviation
    
    # Customer segmentation
    business_proportion: float = 0.30  # 30% business, 70% leisure
    
    # WTP distributions (by segment)
    business_wtp_mean: float = 800.0
    business_wtp_std: float = 200.0
    leisure_wtp_mean: float = 300.0
    leisure_wtp_std: float = 100.0
    
    # Booking curve (demand by days to departure)
    booking_curve: Optional[Dict[int, float]] = None  # {dtd: multiplier}
    
    # Temporal patterns
    seasonality: Optional[Dict[int, float]] = None  # {month: multiplier}
    day_of_week_pattern: Optional[Dict[int, float]] = None  # {0-6: multiplier}
    holidays: Optional[Dict[date, float]] = None  # {date: multiplier}
    
    # Advanced booking window
    mean_advance_purchase: float = 21.0  # Days
    advance_purchase_std: float = 14.0
    min_advance_days: int = 0
    max_advance_days: int = 365
    
    # Party size distribution
    mean_party_size: float = 1.2
    max_party_size: int = 9
    
    # Preferred cabin
    first_proportion: float = 0.05
    business_proportion_cabin: float = 0.20
    premium_economy_proportion: float = 0.15
    economy_proportion: float = 0.60
    
    # Channel distribution
    direct_online_proportion: float = 0.40
    direct_mobile_proportion: float = 0.25
    ota_proportion: float = 0.20
    gds_proportion: float = 0.10
    call_center_proportion: float = 0.05

    def __post_init__(self):
        """Initialize default patterns if not provided."""
        if self.seasonality is None:
            self.seasonality = self.get_default_seasonality()
        if self.day_of_week_pattern is None:
            self.day_of_week_pattern = self.get_default_dow_pattern()
        if self.booking_curve is None:
            self.booking_curve = self.get_default_booking_curve()
    
    def get_default_seasonality(self) -> Dict[int, float]:
        """
        Generate default monthly seasonality factors.
        
        Based on typical northern hemisphere travel patterns:
        - Summer peak (Jun-Aug)
        - Winter holiday peak (Dec)
        - Shoulder seasons (Jan-Feb, Sep-Nov)
        """
        return {
            1: 0.85,  # Jan: Post-holiday slump
            2: 0.90,  # Feb: Low season
            3: 1.00,  # Mar: Spring break start
            4: 1.05,  # Apr: Spring
            5: 1.05,  # May: Pre-summer
            6: 1.20,  # Jun: Summer start
            7: 1.30,  # Jul: Summer peak
            8: 1.30,  # Aug: Summer peak
            9: 1.00,  # Sep: Back to school
            10: 1.00, # Oct: Fall
            11: 0.95, # Nov: Pre-holiday (excluding Thanksgiving)
            12: 1.25  # Dec: Holiday season
        }

    def get_default_dow_pattern(self) -> Dict[int, float]:
        """
        Generate default day-of-week demand multipliers.
        0=Monday, 6=Sunday.
        """
        return {
            0: 1.10,  # Mon: High business demand
            1: 0.90,  # Tue: Low demand
            2: 0.90,  # Wed: Low demand
            3: 1.00,  # Thu: Moderate demand
            4: 1.25,  # Fri: High business/leisure
            5: 0.85,  # Sat: Low business, moderate leisure
            6: 1.15   # Sun: High leisure return
        }
    
    def get_default_booking_curve(self) -> Dict[int, float]:
        """
        Generate default booking curve.
        
        Demand typically peaks around 2-3 weeks before departure
        for leisure, closer for business.
        """
        if self.booking_curve is not None:
            return self.booking_curve
        
        curve = {}
        for dtd in range(0, 91):  # 0 to 90 days
            if dtd <= 1:
                # Very little demand in last 1-2 days
                curve[dtd] = 0.2
            elif dtd <= 7:
                # Increasing as we get closer
                curve[dtd] = 0.5 + (7 - dtd) * 0.1
            elif dtd <= 21:
                # Peak booking period
                curve[dtd] = 1.2
            elif dtd <= 45:
                # Normal demand
                curve[dtd] = 1.0
            elif dtd <= 60:
                # Lower demand far out
                curve[dtd] = 0.7
            else:
                # Very low demand > 60 days out
                curve[dtd] = 0.4
        
        return curve


class DemandGenerator:
    """
    Generates realistic booking requests for a demand stream.
    
    Uses:
    - Poisson process for arrivals
    - Log-normal distributions for WTP
    - Realistic booking curves
    - Customer segmentation
    """
    
    def __init__(
        self,
        config: DemandStreamConfig,
        random_seed: Optional[int] = None
    ):
        """
        Initialize demand generator.
        
        Args:
            config: Demand stream configuration
            random_seed: Random seed for reproducibility
        """
        self.config = config
        self.logger = logging.getLogger(f'DemandGenerator.{config.stream_id}')
        
        # Random state
        self.rng = np.random.default_rng(random_seed)
        
        # Statistics
        self.requests_generated = 0
        self.requests_by_segment: Dict[CustomerSegment, int] = {
            seg: 0 for seg in CustomerSegment
        }
    
    def generate_requests(
        self,
        start_date: date,
        end_date: date,
        market_conditions: Optional[MarketConditions] = None
    ) -> List[BookingRequest]:
        """
        Generate all booking requests for date range.
        
        Args:
            start_date: Start of simulation period (departure dates)
            end_date: End of simulation period (departure dates)
            market_conditions: Optional market conditions affecting demand
            
        Returns:
            List of booking requests with timestamps
        """
        requests = []
        current_departure_date = start_date
        
        while current_departure_date <= end_date:
            # Generate requests for this departure date
            departure_requests = self._generate_requests_for_departure(
                current_departure_date, 
                start_date,
                market_conditions
            )
            requests.extend(departure_requests)
            current_departure_date += timedelta(days=1)
        
        self.logger.info(
            f"Generated {len(requests)} requests for {self.config.stream_id}"
        )
        
        return requests
    
    def _generate_requests_for_departure(
        self,
        departure_date: date,
        simulation_start_date: date,
        market_conditions: Optional[MarketConditions] = None
    ) -> List[BookingRequest]:
        """Generate requests for a single departure date."""
        
        # Determine expected demand for this departure date
        base_demand_mean = self.config.mean_daily_demand
        base_demand_std = self.config.demand_std
        
        # Apply day-of-week pattern
        if self.config.day_of_week_pattern:
            dow_multiplier = self.config.day_of_week_pattern.get(
                departure_date.weekday(), 1.0
            )
            base_demand_mean *= dow_multiplier
            base_demand_std *= dow_multiplier
        
        # Apply seasonality
        if self.config.seasonality:
            season_multiplier = self.config.seasonality.get(
                departure_date.month, 1.0
            )
            base_demand_mean *= season_multiplier
            base_demand_std *= season_multiplier
            
        # Apply holidays
        if self.config.holidays:
            holiday_multiplier = self.config.holidays.get(departure_date, 1.0)
            base_demand_mean *= holiday_multiplier
            base_demand_std *= holiday_multiplier
        
        # Apply market conditions
        if market_conditions:
            base_demand_mean *= market_conditions.seasonality_factor
            base_demand_std *= market_conditions.seasonality_factor
        
        # Sample actual total demand (Normal Distribution)
        # Ensure non-negative
        total_demand = max(0, int(self.rng.normal(base_demand_mean, base_demand_std)))
        
        requests = []
        for _ in range(total_demand):
            request = self._generate_single_request_for_departure(
                departure_date, 
                simulation_start_date
            )
            if request:
                requests.append(request)
        
        return requests
    
    def _generate_single_request_for_departure(
        self,
        departure_date: date,
        simulation_start_date: date
    ) -> Optional[BookingRequest]:
        """Generate a single booking request for a specific departure."""
        
        # Determine customer segment
        segment = self._sample_customer_segment()
        
        # Sample advance purchase days (booking curve)
        dtd = self._sample_dtd_from_curve(segment)
        
        # Calculate booking date
        booking_date = departure_date - timedelta(days=dtd)
        
        # Generate customer
        customer = self._generate_customer(segment, dtd)
        
        # Generate request timestamp (random time during booking day)
        hour = self.rng.integers(0, 24)
        minute = self.rng.integers(0, 60)
        second = self.rng.integers(0, 60)
        request_time = datetime.combine(
            booking_date, 
            time(hour, minute, second)
        )
        
        # Preferred departure time (business travelers more specific)
        if segment == CustomerSegment.BUSINESS:
            # Business prefer morning or evening
            if self.rng.random() < 0.6:
                pref_hour = self.rng.choice([6, 7, 8, 17, 18, 19])
            else:
                pref_hour = self.rng.integers(6, 22)
        else:
            # Leisure more flexible
            pref_hour = self.rng.integers(6, 22) if self.rng.random() < 0.5 else None
        
        preferred_departure_time = time(pref_hour, 0) if pref_hour else None
        
        # Party size
        party_size = self._sample_party_size(segment)
        
        # Preferred cabin
        preferred_cabin = self._sample_preferred_cabin(segment)
        
        # Trip type
        trip_type = self._sample_trip_type(segment)
        
        # Channel
        channel = self._sample_booking_channel(segment)
        
        # Create request
        request = BookingRequest(
            request_time=request_time,
            customer=customer,
            origin=self.config.origin,
            destination=self.config.destination,
            departure_date=departure_date,
            preferred_departure_time=preferred_departure_time,
            party_size=party_size,
            preferred_cabin=preferred_cabin,
            trip_type=trip_type,
            channel=channel
        )
        
        self.requests_generated += 1
        self.requests_by_segment[segment] += 1
        
        return request

    def _sample_dtd_from_curve(self, segment: CustomerSegment) -> int:
        """Sample Days To Departure based on booking curve and segment."""
        
        if segment == CustomerSegment.BUSINESS:
            # Business books late (0-14 days)
            mean = 7.0
            std = 5.0
        elif segment == CustomerSegment.LEISURE:
            # Leisure books early (21-60 days)
            mean = 45.0
            std = 20.0
        elif segment == CustomerSegment.PREMIUM_LEISURE:
            # Premium Leisure books moderately early (14-45 days)
            mean = 30.0
            std = 15.0
        elif segment == CustomerSegment.VFR:
            # VFR books very early or very late (bimodal), simplified to early
            mean = 60.0
            std = 30.0
        else: # Group
            mean = 90.0
            std = 30.0
            
        advance_days = self.rng.normal(mean, std)
        
        return int(np.clip(
            advance_days,
            self.config.min_advance_days,
            self.config.max_advance_days
        ))
        
        return request
    
    def _sample_customer_segment(self) -> CustomerSegment:
        """Sample customer segment."""
        rand = self.rng.random()
        
        # Default proportions if not specified in config
        # Business: 30%, Leisure: 40%, Premium Leisure: 15%, VFR: 10%, Group: 5%
        
        # Use config business proportion as anchor
        biz_prop = self.config.business_proportion
        remaining = 1.0 - biz_prop
        
        if rand < biz_prop:
            return CustomerSegment.BUSINESS
        elif rand < biz_prop + (remaining * 0.5): # 50% of remaining is Leisure
            return CustomerSegment.LEISURE
        elif rand < biz_prop + (remaining * 0.75): # 25% of remaining is Premium Leisure
            return CustomerSegment.PREMIUM_LEISURE
        elif rand < biz_prop + (remaining * 0.9): # 15% of remaining is VFR
            return CustomerSegment.VFR
        else:
            return CustomerSegment.GROUP
    
    def _sample_departure_date(
        self,
        booking_date: date,
        end_date: date
    ) -> Optional[date]:
        """Sample departure date based on advance purchase distribution."""
        
        # Sample advance purchase days (truncated normal)
        advance_days = self.rng.normal(
            self.config.mean_advance_purchase,
            self.config.advance_purchase_std
        )
        
        # Clip to valid range
        advance_days = int(np.clip(
            advance_days,
            self.config.min_advance_days,
            self.config.max_advance_days
        ))
        
        departure_date = booking_date + timedelta(days=advance_days)
        
        # Must be within simulation period
        if departure_date > end_date:
            return None
        
        return departure_date
    
    def _generate_customer(
        self,
        segment: CustomerSegment,
        dtd: int
    ) -> Customer:
        """Generate customer with attributes."""
        
        # Willingness to pay (log-normal distribution)
        if segment == CustomerSegment.BUSINESS:
            mean_wtp = self.config.business_wtp_mean
            std_wtp = self.config.business_wtp_std
        elif segment == CustomerSegment.PREMIUM_LEISURE:
            mean_wtp = self.config.leisure_wtp_mean * 1.5
            std_wtp = self.config.leisure_wtp_std * 1.2
        elif segment == CustomerSegment.VFR:
            mean_wtp = self.config.leisure_wtp_mean * 0.8
            std_wtp = self.config.leisure_wtp_std
        elif segment == CustomerSegment.GROUP:
            mean_wtp = self.config.leisure_wtp_mean * 0.7
            std_wtp = self.config.leisure_wtp_std * 0.5
        else: # LEISURE
            mean_wtp = self.config.leisure_wtp_mean
            std_wtp = self.config.leisure_wtp_std
            
        wtp = self.rng.lognormal(
            mean=np.log(mean_wtp),
            sigma=std_wtp / mean_wtp
        )
        
        # Price sensitivity (leisure more price sensitive)
        if segment == CustomerSegment.BUSINESS:
            price_sensitivity = self.rng.uniform(0.5, 1.0)
            time_sensitivity = self.rng.uniform(1.0, 1.5)
        elif segment == CustomerSegment.PREMIUM_LEISURE:
            price_sensitivity = self.rng.uniform(0.8, 1.2)
            time_sensitivity = self.rng.uniform(0.8, 1.2)
        else:
            price_sensitivity = self.rng.uniform(1.0, 1.8)
            time_sensitivity = self.rng.uniform(0.5, 1.0)
        
        # Loyalty (random)
        loyalty_score = self.rng.beta(2, 5)  # Skewed toward low loyalty
        
        customer = Customer(
            segment=segment,
            willingness_to_pay=wtp,
            advance_purchase_days=dtd,
            price_sensitivity=price_sensitivity,
            time_sensitivity=time_sensitivity,
            loyalty_score=loyalty_score
        )
        
        return customer
    
    def _sample_party_size(self, segment: CustomerSegment) -> int:
        """Sample party size."""
        if segment == CustomerSegment.BUSINESS:
            # Business mostly travel alone
            return 1 if self.rng.random() < 0.9 else 2
        else:
            # Leisure vary more
            size = self.rng.poisson(self.config.mean_party_size)
            return max(1, min(size, self.config.max_party_size))
    
    def _sample_preferred_cabin(self, segment: CustomerSegment) -> CabinClass:
        """Sample preferred cabin class."""
        
        cabins = [CabinClass.FIRST, CabinClass.BUSINESS, 
                  CabinClass.PREMIUM_ECONOMY, CabinClass.ECONOMY]

        if segment == CustomerSegment.BUSINESS:
            probs = [0.10, 0.40, 0.20, 0.30]  # F, J, W, Y
        elif segment == CustomerSegment.PREMIUM_LEISURE:
            probs = [0.05, 0.15, 0.40, 0.40]
        elif segment == CustomerSegment.VFR:
            probs = [0.01, 0.04, 0.05, 0.90]
        else: # Leisure & Group
            probs = [0.02, 0.08, 0.10, 0.80]
        
        return self.rng.choice(cabins, p=probs)
    
    def _sample_trip_type(self, segment: CustomerSegment) -> TripType:
        """Sample trip type."""
        if segment == CustomerSegment.BUSINESS:
            # Business often one-way
            return self.rng.choice(
                [TripType.ONE_WAY, TripType.ROUND_TRIP],
                p=[0.6, 0.4]
            )
        else:
            # Leisure mostly round-trip
            return self.rng.choice(
                [TripType.ONE_WAY, TripType.ROUND_TRIP],
                p=[0.2, 0.8]
            )
    
    def _sample_booking_channel(self, segment: CustomerSegment) -> BookingChannel:
        """Sample booking channel."""
        
        if segment == CustomerSegment.BUSINESS:
            # Business use more GDS and corporate tools
            channels = [
                BookingChannel.DIRECT_ONLINE,
                BookingChannel.DIRECT_MOBILE,
                BookingChannel.GDS,
                BookingChannel.CORPORATE,
                BookingChannel.CALL_CENTER
            ]
            probs = [0.30, 0.20, 0.25, 0.20, 0.05]
        elif segment == CustomerSegment.PREMIUM_LEISURE:
             channels = [
                BookingChannel.DIRECT_ONLINE,
                BookingChannel.DIRECT_MOBILE,
                BookingChannel.OTA,
                BookingChannel.CALL_CENTER
            ]
             probs = [0.40, 0.30, 0.25, 0.05]
        else:
            # Leisure use more OTA
            channels = [
                BookingChannel.DIRECT_ONLINE,
                BookingChannel.DIRECT_MOBILE,
                BookingChannel.OTA,
                BookingChannel.CALL_CENTER
            ]
            probs = [0.35, 0.30, 0.30, 0.05]
        
        return self.rng.choice(channels, p=probs)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get generation statistics."""
        return {
            'stream_id': self.config.stream_id,
            'total_requests': self.requests_generated,
            'by_segment': dict(self.requests_by_segment),
            'mean_daily_demand': self.config.mean_daily_demand
        }


class MultiStreamDemandGenerator:
    """
    Manages multiple demand streams and generates all requests.
    
    This coordinates demand generation across multiple O-D pairs.
    """
    
    def __init__(
        self,
        stream_configs: List[DemandStreamConfig],
        random_seed: Optional[int] = None
    ):
        """
        Initialize multi-stream generator.
        
        Args:
            stream_configs: List of demand stream configurations
            random_seed: Base random seed
        """
        self.stream_configs = stream_configs
        self.logger = logging.getLogger('MultiStreamDemandGenerator')
        
        # Create generator for each stream
        self.generators: Dict[str, DemandGenerator] = {}
        for i, config in enumerate(stream_configs):
            seed = random_seed + i if random_seed else None
            self.generators[config.stream_id] = DemandGenerator(config, seed)
    
    def generate_all_requests(
        self,
        start_date: date,
        end_date: date,
        market_conditions: Optional[MarketConditions] = None
    ) -> List[BookingRequest]:
        """
        Generate requests for all streams.
        
        Args:
            start_date: Simulation start date
            end_date: Simulation end date
            market_conditions: Market conditions
            
        Returns:
            Combined list of all booking requests
        """
        all_requests = []
        
        for stream_id, generator in self.generators.items():
            self.logger.info(f"Generating demand for stream: {stream_id}")
            requests = generator.generate_requests(
                start_date,
                end_date,
                market_conditions
            )
            all_requests.extend(requests)
        
        # Sort by request time
        all_requests.sort(key=lambda r: r.request_time)
        
        self.logger.info(f"Generated {len(all_requests)} total requests")
        
        return all_requests
    
    def add_requests_to_event_queue(
        self,
        event_manager: EventManager,
        requests: List[BookingRequest]
    ) -> int:
        """
        Add booking requests to event queue.
        
        Args:
            event_manager: Event manager to add to
            requests: List of booking requests
            
        Returns:
            Number of events added
        """
        for request in requests:
            # Determine which stream this request came from
            stream_id = f"{request.origin.code}-{request.destination.code}"
            
            event_manager.create_and_add_event(
                timestamp=request.request_time,
                event_type=EventType.BOOKING_REQUEST,
                data=BookingRequestEvent(
                    request=request,
                    demand_stream_id=stream_id
                ),
                priority=2  # Normal priority
            )
        
        return len(requests)
    
    def get_total_statistics(self) -> Dict[str, any]:
        """Get combined statistics from all streams."""
        stats = {
            'num_streams': len(self.generators),
            'streams': {}
        }
        
        total_requests = 0
        for stream_id, generator in self.generators.items():
            stream_stats = generator.get_statistics()
            stats['streams'][stream_id] = stream_stats
            total_requests += stream_stats['total_requests']
        
        stats['total_requests'] = total_requests
        
        return stats


def generate_default_holidays(years: List[int]) -> Dict[date, float]:
    """
    Generate default holiday multipliers for given years.
    
    Includes:
    - New Year's (Low on day, High after)
    - Valentine's Day (Minor bump)
    - July 4th (US Independence Day)
    - Christmas Season (High before/after, Low on day)
    """
    holidays = {}
    for year in years:
        # New Year's
        holidays[date(year, 1, 1)] = 0.6  # Low demand on actual day
        holidays[date(year, 1, 2)] = 1.4  # Return travel
        holidays[date(year, 1, 3)] = 1.3
        
        # Valentine's (minor bump)
        holidays[date(year, 2, 14)] = 1.1
        
        # Easter (Western) - Computus Algorithm
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        easter_sunday = date(year, month, day)
        
        holidays[easter_sunday] = 0.8  # Low travel on Sunday itself
        holidays[easter_sunday - timedelta(days=2)] = 1.3  # Good Friday
        holidays[easter_sunday + timedelta(days=1)] = 1.2  # Easter Monday
        
        # Memorial Day (US) - Last Monday of May
        may_31 = date(year, 5, 31)
        # weekday(): Mon=0. Days to subtract to get to Mon: may_31.weekday()
        memorial_day = may_31 - timedelta(days=may_31.weekday())
        holidays[memorial_day] = 0.7
        holidays[memorial_day - timedelta(days=3)] = 1.3  # Friday before
        
        # July 4th (US)
        holidays[date(year, 7, 4)] = 0.7
        holidays[date(year, 7, 3)] = 1.2
        holidays[date(year, 7, 5)] = 1.2
        
        # Labor Day (US) - 1st Monday of September
        sep_1 = date(year, 9, 1)
        days_to_mon = (0 - sep_1.weekday() + 7) % 7
        labor_day = sep_1 + timedelta(days=days_to_mon)
        holidays[labor_day] = 0.7
        holidays[labor_day - timedelta(days=3)] = 1.3 # Friday before
        
        # Thanksgiving (US) - 4th Thursday of November
        # Calculate 4th Thursday
        nov_1 = date(year, 11, 1)
        # weekday(): Mon=0, Thu=3
        # Days to first Thursday: (3 - nov_1.weekday() + 7) % 7
        days_to_first_thu = (3 - nov_1.weekday() + 7) % 7
        first_thu = nov_1 + timedelta(days=days_to_first_thu)
        thanksgiving = first_thu + timedelta(weeks=3)
        
        holidays[thanksgiving] = 0.6  # Low demand on actual day
        holidays[thanksgiving - timedelta(days=1)] = 1.5  # Wed before (Peak travel)
        holidays[thanksgiving + timedelta(days=3)] = 1.5  # Sun after (Peak return)
        
        # Christmas Season
        # Pre-holiday rush
        for d in range(20, 25):
            try:
                holidays[date(year, 12, d)] = 1.4
            except ValueError:
                pass
                
        # Christmas Day
        holidays[date(year, 12, 25)] = 0.5
        
        # Post-holiday / Pre-NYE
        for d in range(26, 31):
            try:
                holidays[date(year, 12, d)] = 1.2
            except ValueError:
                pass
            
    return holidays
