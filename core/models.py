"""
Core data models for the airline revenue management simulator.

This module defines the fundamental data structures representing flights,
bookings, inventory, customers, and other core entities.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, time, timedelta
from enum import Enum
from typing import Optional, List, Dict, Tuple
from decimal import Decimal
import uuid


class CabinClass(Enum):
    """Aircraft cabin classes."""
    FIRST = "F"
    BUSINESS = "J"
    PREMIUM_ECONOMY = "W"
    ECONOMY = "Y"


class BookingClass(Enum):
    """Booking fare classes (nested within cabins)."""
    # First class
    F = "F"
    A = "A"
    
    # Business class
    J = "J"
    C = "C"
    D = "D"
    I = "I"
    
    # Premium Economy
    W = "W"
    E = "E"
    
    # Economy
    Y = "Y"
    B = "B"
    M = "M"
    H = "H"
    Q = "Q"
    K = "K"
    L = "L"
    
    def get_cabin(self) -> CabinClass:
        """Return the cabin this booking class belongs to."""
        if self.value in ['F', 'A']:
            return CabinClass.FIRST
        elif self.value in ['J', 'C', 'D', 'I']:
            return CabinClass.BUSINESS
        elif self.value in ['W', 'E']:
            return CabinClass.PREMIUM_ECONOMY
        else:
            return CabinClass.ECONOMY


class CustomerSegment(Enum):
    """Customer market segments."""
    BUSINESS = "business"
    LEISURE = "leisure"
    VFR = "vfr"  # Visiting Friends and Relatives
    GROUP = "group"


class TripType(Enum):
    """Type of trip."""
    ONE_WAY = "oneway"
    ROUND_TRIP = "roundtrip"
    MULTI_CITY = "multicity"


class BookingChannel(Enum):
    """Distribution channel."""
    DIRECT_ONLINE = "direct_online"
    DIRECT_MOBILE = "direct_mobile"
    CALL_CENTER = "call_center"
    OTA = "ota"  # Online Travel Agency
    GDS = "gds"  # Global Distribution System
    CORPORATE = "corporate"


class EventType(Enum):
    """Simulation event types."""
    BOOKING_REQUEST = "booking_request"
    CANCELLATION = "cancellation"
    NO_SHOW = "no_show"
    RM_OPTIMIZATION = "rm_optimization"
    PRICE_UPDATE = "price_update"
    SNAPSHOT = "snapshot"
    COMPETITOR_ACTION = "competitor_action"


@dataclass
class Airport:
    """Airport definition."""
    code: str  # IATA code (e.g., "JFK")
    name: str
    city: str
    country: str
    timezone: str
    lat: float
    lon: float
    
    def __str__(self) -> str:
        return f"{self.code} ({self.city})"
    
    def __hash__(self) -> int:
        return hash(self.code)


@dataclass
class Route:
    """Flight route (origin-destination pair)."""
    origin: Airport
    destination: Airport
    distance_km: float
    
    def __str__(self) -> str:
        return f"{self.origin.code}-{self.destination.code}"
    
    def __hash__(self) -> int:
        return hash((self.origin.code, self.destination.code))
    
    @property
    def reverse(self) -> 'Route':
        """Get the reverse route."""
        return Route(self.destination, self.origin, self.distance_km)


@dataclass
class Aircraft:
    """Aircraft type definition."""
    type_code: str  # e.g., "B777", "A320"
    name: str
    capacity: Dict[CabinClass, int]  # Seats by cabin
    
    @property
    def total_capacity(self) -> int:
        """Total aircraft capacity."""
        return sum(self.capacity.values())
    
    def __str__(self) -> str:
        return f"{self.type_code} ({self.total_capacity} seats)"


@dataclass
class FlightSchedule:
    """Scheduled flight information."""
    airline_code: str
    flight_number: str
    route: Route
    departure_time: time
    arrival_time: time  # Local times
    days_of_week: List[int]  # 0=Monday, 6=Sunday
    aircraft: Aircraft
    valid_from: date
    valid_until: date
    
    @property
    def flight_code(self) -> str:
        """Full flight code (e.g., 'AA100')."""
        return f"{self.airline_code}{self.flight_number}"
    
    def operates_on(self, date_obj: date) -> bool:
        """Check if flight operates on given date."""
        if not (self.valid_from <= date_obj <= self.valid_until):
            return False
        return date_obj.weekday() in self.days_of_week
    
    def __str__(self) -> str:
        return f"{self.flight_code} {self.route} {self.departure_time}"


@dataclass
class FlightDate:
    """Specific flight instance on a particular date."""
    schedule: FlightSchedule
    departure_date: date
    
    # Inventory state
    capacity: Dict[CabinClass, int]
    bookings: Dict[BookingClass, int] = field(default_factory=dict)
    
    # RM controls
    booking_limits: Dict[BookingClass, int] = field(default_factory=dict)
    bid_prices: List[float] = field(default_factory=list)
    
    # Operational status
    is_closed: bool = False
    actual_departure: Optional[datetime] = None
    
    @property
    def flight_id(self) -> str:
        """Unique identifier for this flight instance."""
        return f"{self.schedule.flight_code}_{self.departure_date.isoformat()}"
    
    @property
    def departure_datetime(self) -> datetime:
        """Full departure datetime."""
        return datetime.combine(self.departure_date, self.schedule.departure_time)
    
    @property
    def arrival_datetime(self) -> datetime:
        """Full arrival datetime (may be next day)."""
        dt = datetime.combine(self.departure_date, self.schedule.arrival_time)
        # Handle overnight flights
        if self.schedule.arrival_time < self.schedule.departure_time:
            dt += timedelta(days=1)
        return dt
    
    def total_bookings(self, cabin: Optional[CabinClass] = None) -> int:
        """Total bookings, optionally filtered by cabin."""
        if cabin is None:
            return sum(self.bookings.values())
        return sum(count for bc, count in self.bookings.items() 
                  if bc.get_cabin() == cabin)
    
    def load_factor(self, cabin: Optional[CabinClass] = None) -> float:
        """Current load factor (0.0 to 1.0)."""
        if cabin is None:
            total_cap = sum(self.capacity.values())
            return self.total_bookings() / total_cap if total_cap > 0 else 0.0
        
        cabin_cap = self.capacity.get(cabin, 0)
        return self.total_bookings(cabin) / cabin_cap if cabin_cap > 0 else 0.0
    
    def available_seats(self, booking_class: BookingClass) -> int:
        """Calculate available seats for a booking class."""
        cabin = booking_class.get_cabin()
        
        # Return 0 if cabin doesn't exist on this aircraft
        if cabin not in self.capacity:
            return 0
            
        cabin_capacity = self.capacity[cabin]
        cabin_bookings = self.total_bookings(cabin)
        physical_available = cabin_capacity - cabin_bookings
        
        # Apply booking limit
        class_limit = self.booking_limits.get(booking_class, cabin_capacity)
        class_bookings = self.bookings.get(booking_class, 0)
        limit_available = class_limit - class_bookings
        
        return max(0, min(physical_available, limit_available))
    
    def __str__(self) -> str:
        return f"{self.schedule.flight_code} on {self.departure_date}"


@dataclass
class Customer:
    """Customer making a booking."""
    customer_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    segment: CustomerSegment = CustomerSegment.LEISURE
    
    # Preferences
    willingness_to_pay: float = 0.0
    preferred_airline: Optional[str] = None
    preferred_departure_time: Optional[time] = None
    time_sensitivity: float = 1.0  # How much they value schedule
    price_sensitivity: float = 1.0  # How much they value price
    
    # Booking behavior
    advance_purchase_days: int = 0
    flexibility_days: int = 0  # How many days flexible on travel date
    
    # Loyalty
    frequent_flyer_tier: Optional[str] = None
    loyalty_score: float = 0.0  # 0.0 to 1.0
    
    def __str__(self) -> str:
        return f"Customer({self.segment.value}, WTP=${self.willingness_to_pay:.0f})"


@dataclass
class BookingRequest:
    """Customer booking request."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    request_time: datetime = field(default_factory=datetime.now)
    
    # Customer
    customer: Customer = field(default_factory=Customer)
    
    # Trip details
    origin: Airport = None
    destination: Airport = None
    departure_date: date = None
    preferred_departure_time: Optional[time] = None
    return_date: Optional[date] = None  # For round trips
    
    # Party details
    party_size: int = 1
    
    # Preferences
    preferred_cabin: CabinClass = CabinClass.ECONOMY
    trip_type: TripType = TripType.ONE_WAY
    channel: BookingChannel = BookingChannel.DIRECT_ONLINE
    
    @property
    def days_to_departure(self) -> int:
        """Days from request to departure."""
        if self.departure_date is None:
            return 0
        return (self.departure_date - self.request_time.date()).days
    
    def __str__(self) -> str:
        return (f"Request {self.origin.code}->{self.destination.code} "
                f"on {self.departure_date}, Party={self.party_size}")


@dataclass
class TravelSolution:
    """A travel option (one or more flight segments)."""
    solution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Flight segments (direct or connecting)
    flights: List[FlightDate] = field(default_factory=list)
    booking_classes: List[BookingClass] = field(default_factory=list)
    
    # Pricing
    base_fare: float = 0.0
    taxes_fees: float = 0.0
    total_price: float = 0.0
    
    # Availability
    available_seats: int = 0
    
    # Attributes for choice modeling
    total_travel_time: timedelta = timedelta()
    num_connections: int = 0
    
    @property
    def is_direct(self) -> bool:
        """Check if solution is a direct flight."""
        return len(self.flights) == 1
    
    @property
    def origin(self) -> Airport:
        """Origin airport."""
        return self.flights[0].schedule.route.origin
    
    @property
    def destination(self) -> Airport:
        """Final destination airport."""
        return self.flights[-1].schedule.route.destination
    
    @property
    def departure_time(self) -> datetime:
        """First flight departure time."""
        return self.flights[0].departure_datetime
    
    @property
    def arrival_time(self) -> datetime:
        """Last flight arrival time."""
        return self.flights[-1].arrival_datetime
    
    def __str__(self) -> str:
        flight_codes = "->".join(f.schedule.flight_code for f in self.flights)
        return f"{flight_codes} ${self.total_price:.0f}"


@dataclass
class Booking:
    """Confirmed booking."""
    booking_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    booking_time: datetime = field(default_factory=datetime.now)
    
    # Customer and request
    customer: Customer = None
    original_request: BookingRequest = None
    
    # Booked solution
    solution: TravelSolution = None
    party_size: int = 1
    
    # Financial
    total_revenue: float = 0.0
    payment_method: str = "card"
    currency: str = "USD"
    exchange_rate: float = 1.0
    
    # PSS Reference
    record_locator: Optional[str] = None
    
    # Status
    is_cancelled: bool = False
    cancellation_time: Optional[datetime] = None
    is_no_show: bool = False
    
    # Ancillary revenue
    ancillary_revenue: float = 0.0
    
    @property
    def revenue_per_passenger(self) -> float:
        """Revenue per passenger in party."""
        return self.total_revenue / self.party_size if self.party_size > 0 else 0.0
    
    def __str__(self) -> str:
        status = "Cancelled" if self.is_cancelled else "Active"
        return f"Booking {self.booking_id[:8]} ({status}): {self.solution}"


@dataclass
class Fare:
    """Fare product definition."""
    fare_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Route and class
    origin: Airport = None
    destination: Airport = None
    booking_class: BookingClass = BookingClass.Y
    
    # Pricing
    base_fare: float = 0.0
    currency: str = "USD"
    
    # Rules
    advance_purchase_days: int = 0  # Min days before departure
    min_stay_days: int = 0
    max_stay_days: int = 365
    saturday_night_stay_required: bool = False
    refundable: bool = True
    changeable: bool = True
    change_fee: float = 0.0
    
    # Validity
    valid_from: date = field(default_factory=date.today)
    valid_until: date = field(default_factory=lambda: date.today() + timedelta(days=365))
    
    # Inventory control
    availability_count: Optional[int] = None  # Inventory allocated to this fare
    
    def is_valid(self, booking_date: date, departure_date: date) -> bool:
        """Check if fare is valid for given dates."""
        if not (self.valid_from <= booking_date <= self.valid_until):
            return False
        
        days_advance = (departure_date - booking_date).days
        if days_advance < self.advance_purchase_days:
            return False
        
        return True
    
    def __str__(self) -> str:
        return (f"{self.booking_class.value} {self.origin.code}-{self.destination.code} "
                f"${self.base_fare:.0f}")


@dataclass
class RMControl:
    """Revenue management control settings for a flight."""
    flight_date: FlightDate = None
    last_optimization_time: Optional[datetime] = None
    
    # Nested booking limits
    booking_limits: Dict[BookingClass, int] = field(default_factory=dict)
    protection_levels: Dict[BookingClass, int] = field(default_factory=dict)
    
    # Bid price vector (one per seat)
    bid_prices: List[float] = field(default_factory=list)
    
    # Forecasts
    demand_forecast: Dict[BookingClass, Tuple[float, float]] = field(default_factory=dict)  # (mean, std)
    
    # Algorithm metadata
    optimization_method: str = "EMSR-b"
    optimization_duration_ms: float = 0.0
    
    def __str__(self) -> str:
        return f"RM Controls for {self.flight_date} (Method: {self.optimization_method})"


@dataclass
class MarketConditions:
    """Current market conditions affecting demand and pricing."""
    date: date = field(default_factory=date.today)
    
    # Economic factors
    gdp_growth_rate: float = 0.02  # Annual
    unemployment_rate: float = 0.05
    consumer_confidence_index: float = 100.0
    
    # Industry factors
    fuel_price_index: float = 100.0
    average_fare_index: float = 100.0
    capacity_utilization: float = 0.80
    
    # Seasonal factors
    seasonality_factor: float = 1.0  # Multiplicative factor
    holiday_indicator: bool = False
    special_event: Optional[str] = None
    
    # Competition
    market_concentration: float = 0.5  # HHI-like measure
    avg_competitor_price: float = 0.0
    
    def __str__(self) -> str:
        return f"Market {self.date}: Seasonality={self.seasonality_factor:.2f}"


# Type aliases for clarity
Price = float
Quantity = int
Probability = float
Revenue = float
