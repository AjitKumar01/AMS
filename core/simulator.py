"""
Main simulation engine orchestrating all components.

This is the core simulator that coordinates:
- Event processing
- Demand generation
- Inventory management
- Revenue management optimization
- Customer choice modeling
- Competitive dynamics
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import List, Dict, Optional, Any, Callable, Union
from enum import Enum
import logging
from tqdm import tqdm
import numpy as np

from core.models import (
    FlightSchedule, FlightDate, Airport, Route, BookingRequest,
    Booking, TravelSolution, EventType, CabinClass, BookingClass,
    Customer, MarketConditions, CustomerSegment
)
from core.events import (
    EventManager, EventScheduler, Event, BookingRequestEvent,
    CancellationEvent, RMOptimizationEvent, SnapshotEvent,
    EventPriority, ProgressTracker
)
from rm.optimizer import EMSRbOptimizer, DemandForecast as OptimizerForecast
from demand.forecaster import DemandForecaster, ForecastMethod
from core.reservation import ReservationSystem
from choice.frat5 import FRAT5Model
from core.personalization import PersonalizationEngine


class SimulationMode(Enum):
    """Simulation execution modes."""
    BATCH = "batch"  # Run entire simulation
    STEP = "step"  # Step through events
    REALTIME = "realtime"  # Simulate in real-time
    INTERACTIVE = "interactive"  # With breakpoints


class SimulationStatus(Enum):
    """Current simulation status."""
    INITIALIZED = "initialized"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class SimulationConfig:
    """Configuration for simulation run."""
    
    # Time period
    start_date: date = field(default_factory=date.today)
    end_date: date = field(default_factory=lambda: date.today() + timedelta(days=90))
    
    # Simulation parameters
    random_seed: Optional[int] = 42
    mode: SimulationMode = SimulationMode.BATCH
    
    # RM settings
    rm_optimization_frequency: Union[str, int] = "daily"  # 'daily', 'weekly', or hours (int)
    rm_method: str = "EMSR-b"  # 'EMSR-b', 'DP', 'MC', 'ML'
    optimization_horizons: List[int] = field(default_factory=lambda: [30, 14, 7, 3, 1])  # Days before departure
    
    # Pricing
    dynamic_pricing: bool = True
    price_update_frequency_hours: float = 6.0
    
    # Currency
    customer_currency: str = "USD"
    base_currency: str = "USD"
    currency_rate: float = 1.0
    
    # Demand
    demand_generation_method: str = "poisson"  # 'poisson' or 'stateful'
    forecast_method: str = "pickup" # 'pickup', 'additive_pickup', 'exponential_smoothing'
    
    # Overbooking
    overbooking_enabled: bool = True
    overbooking_method: str = "critical_fractile"  # 'critical_fractile', 'risk_averse'
    overbooking_risk_tolerance: float = 0.05  # Max probability of denied boarding
    
    # Customer choice
    choice_model: str = "mnl"  # 'cheapest', 'mnl', 'enhanced'
    include_buyup_down: bool = True
    include_recapture: bool = True
    
    # Personalization
    personalization_enabled: bool = False
    
    # Database
    use_db: bool = True
    db_path: str = "simulation_results/simulation.db"
    
    # Snapshot settings
    snapshot_frequency_days: int = 7
    snapshot_detailed: bool = False
    
    # Performance
    enable_parallel: bool = False
    num_workers: int = 4
    progress_bar: bool = True
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    
    # Output
    output_dir: str = "simulation_results"
    save_snapshots: bool = True
    save_booking_details: bool = True
    export_csv: bool = True  # Export detailed CSV files
    export_log: bool = True  # Enable file logging


@dataclass
class SimulationResults:
    """Results from a simulation run."""
    
    # Metadata
    config: SimulationConfig = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # Performance metrics
    total_bookings: int = 0
    total_cancellations: int = 0
    total_no_shows: int = 0
    total_revenue: float = 0.0
    
    # By flight
    flights_operated: int = 0
    total_seats_offered: int = 0
    total_seats_sold: int = 0
    
    # Detailed results
    bookings: List[Booking] = field(default_factory=list)
    flight_results: Dict[str, Any] = field(default_factory=dict)
    daily_metrics: Dict[date, Dict[str, float]] = field(default_factory=dict)
    
    # RM performance
    rm_optimization_count: int = 0
    avg_optimization_time_ms: float = 0.0
    
    # Exported files
    exported_files: Dict[str, str] = field(default_factory=dict)
    
    @property
    def duration_seconds(self) -> float:
        """Simulation run duration in seconds."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def load_factor(self) -> float:
        """Overall load factor."""
        if self.total_seats_offered == 0:
            return 0.0
        return self.total_seats_sold / self.total_seats_offered
    
    @property
    def average_fare(self) -> float:
        """Average fare per passenger."""
        if self.total_seats_sold == 0:
            return 0.0
        return self.total_revenue / self.total_seats_sold
    
    @property
    def cancellation_rate(self) -> float:
        """Cancellation rate."""
        if self.total_bookings == 0:
            return 0.0
        return self.total_cancellations / self.total_bookings
    
    def summary(self) -> str:
        """Generate summary report."""
        return f"""
Simulation Results Summary
{'='*50}
Duration: {self.duration_seconds:.1f} seconds
Flights: {self.flights_operated}
Total Revenue: ${self.total_revenue:,.2f}
Total Bookings: {self.total_bookings:,}
Cancellations: {self.total_cancellations} ({self.cancellation_rate:.1%})
Load Factor: {self.load_factor:.1%}
Average Fare: ${self.average_fare:.2f}
RM Optimizations: {self.rm_optimization_count}
"""


class Simulator:
    """
    Main airline revenue management simulator.
    
    This orchestrates all simulation components including:
    - Event management
    - Demand generation  
    - Inventory control
    - Pricing
    - Revenue optimization
    - Customer choice
    """
    
    def __init__(
        self,
        config: SimulationConfig,
        schedules: List[FlightSchedule],
        routes: List[Route],
        airports: List[Airport]
    ):
        """
        Initialize simulator.
        
        Args:
            config: Simulation configuration
            schedules: Flight schedules to simulate
            routes: Routes in the network
            airports: Airports in the network
        """
        self.config = config
        self.schedules = schedules
        self.routes = {(r.origin.code, r.destination.code): r for r in routes}
        self.airports = {a.code: a for a in airports}
        
        # Core components (to be initialized)
        self.event_manager = EventManager()
        self.event_scheduler = EventScheduler(self.event_manager)
        
        # State
        self.flight_dates: Dict[str, FlightDate] = {}
        self.bookings: List[Booking] = []
        self.current_date: date = config.start_date
        self.status = SimulationStatus.INITIALIZED
        
        # Results
        self.results = SimulationResults(config=config)
        
        # Overbooking and choice models
        self.overbooking_optimizer = None
        self.choice_model = None
        
        # Reservation System
        self.reservation_system = ReservationSystem()
        
        # Data export
        self.data_exporter = None
        if config.export_csv:
            from core.data_export import DataExporter
            self.data_exporter = DataExporter(output_dir=config.output_dir)
            
        # Database
        self.db = None
        if config.use_db:
            from core.database import DatabaseManager
            self.db = DatabaseManager(db_path=config.db_path)
        
        # Logging
        self._setup_logging()
        
        # Initialize components (placeholders for now)
        self._initialize_components()
        
    def _setup_logging(self) -> None:
        """Configure logging."""
        level = getattr(logging, self.config.log_level.upper())
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure ROOT logger to capture ALL logging from entire application
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Clear any existing handlers on root logger
        root_logger.handlers = []
        
        # Add console handler to root logger
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # Add file logging if requested
        if self.config.export_log:
            if not self.config.log_file:
                # Auto-generate log filename with timestamp
                from pathlib import Path
                log_dir = Path(self.config.output_dir)
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.config.log_file = str(log_dir / f"simulation_{timestamp}.log")
            
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
            
            self.logger = logging.getLogger('Simulator')
            self.logger.info(f"Logging to file: {self.config.log_file}")
        
        # Create simulator logger (it will inherit root logger's handlers)
        self.logger = logging.getLogger('Simulator')
    
    def _initialize_components(self) -> None:
        """Initialize all simulation components."""
        self.logger.info("Initializing simulation components...")
        
        # Initialize forecaster
        forecast_method = ForecastMethod.PICKUP
        if self.config.forecast_method == "additive_pickup":
            forecast_method = ForecastMethod.ADDITIVE_PICKUP
        elif self.config.forecast_method == "exponential_smoothing":
            forecast_method = ForecastMethod.EXPONENTIAL_SMOOTHING
        elif self.config.forecast_method == "historical_average":
            forecast_method = ForecastMethod.HISTORICAL_AVERAGE
            
        self.forecaster = DemandForecaster(
            method=forecast_method,
            track_accuracy=True
        )
        
        # Initialize RM optimizer
        if self.config.rm_method == "EMSR-b":
            self.rm_optimizer = EMSRbOptimizer()
        elif self.config.rm_method == "BidPrice":
            from rm.bid_price_optimizer import HeuristicBidPriceOptimizer
            self.rm_optimizer = HeuristicBidPriceOptimizer()
            self.logger.info("Using Heuristic Bid Price Optimizer")
        else:
            # Default to EMSR-b if unknown
            self.logger.warning(f"Unknown RM method {self.config.rm_method}, defaulting to EMSR-b")
            self.rm_optimizer = EMSRbOptimizer()
        
        # Initialize overbooking optimizer
        if self.config.overbooking_enabled:
            from overbooking.optimizer import OverbookingOptimizer, OverbookingMethod, NoShowModel
            method = OverbookingMethod.CRITICAL_FRACTILE
            if self.config.overbooking_method == "risk_averse":
                method = OverbookingMethod.RISK_AVERSE
            self.overbooking_optimizer = OverbookingOptimizer(
                no_show_model=NoShowModel(),
                method=method
            )
            self.logger.info(f"Overbooking enabled with {self.config.overbooking_method} method")
        
        # Initialize choice model
        if self.config.choice_model == "mnl":
            from choice.models import MultinomialLogitModel, UtilityFunction
            # Enable Log-Linear Disutility by default for better realism
            utility_fn = UtilityFunction(use_log_price=True)
            self.choice_model = MultinomialLogitModel(utility_function=utility_fn)
            self.logger.info("Customer choice: Multinomial Logit (MNL) with Log-Linear Price Utility")
        elif self.config.choice_model == "enhanced":
            from choice.models import EnhancedChoiceModel
            self.choice_model = EnhancedChoiceModel()
            self.logger.info("Customer choice: Enhanced MNL with buy-up/down and recapture")
        else:
            self.choice_model = None  # Use simple cheapest logic
            self.logger.info("Customer choice: Simple (cheapest within WTP)")
            
        # Initialize FRAT5 model for sell-up logic
        self.frat5_model = FRAT5Model()
        
        # Initialize Personalization Engine
        self.personalization_engine = PersonalizationEngine(
            enabled=self.config.personalization_enabled
        )
        
        # Create flight dates for all schedules
        self._create_flight_dates()
        
        # Register event handlers
        self._register_event_handlers()
        
        # Schedule recurring events
        self._schedule_recurring_events()
        
        self.logger.info(f"Initialized {len(self.flight_dates)} flight dates")
        self.logger.info(f"Scheduled {self.event_manager.size()} initial events")
    
    def _create_flight_dates(self) -> None:
        """Create FlightDate instances for all scheduled flights."""
        current = self.config.start_date
        end = self.config.end_date
        
        while current <= end:
            for schedule in self.schedules:
                if schedule.operates_on(current):
                    flight_date = FlightDate(
                        schedule=schedule,
                        departure_date=current,
                        capacity={cabin: count for cabin, count in schedule.aircraft.capacity.items()}
                    )
                    self.flight_dates[flight_date.flight_id] = flight_date
            
            current += timedelta(days=1)
    
    def _register_event_handlers(self) -> None:
        """Register handlers for different event types."""
        self.event_manager.register_handler(
            EventType.BOOKING_REQUEST, 
            self._handle_booking_request
        )
        self.event_manager.register_handler(
            EventType.CANCELLATION,
            self._handle_cancellation
        )
        self.event_manager.register_handler(
            EventType.RM_OPTIMIZATION,
            self._handle_rm_optimization
        )
        self.event_manager.register_handler(
            EventType.PRICE_UPDATE,
            self._handle_price_update
        )
        self.event_manager.register_handler(
            EventType.SNAPSHOT,
            self._handle_snapshot
        )
    
    def _schedule_recurring_events(self) -> None:
        """Schedule recurring events like RM optimization and snapshots."""
        start_dt = datetime.combine(self.config.start_date, datetime.min.time())
        end_dt = datetime.combine(self.config.end_date, datetime.max.time())
        
        # Schedule RM optimizations
        rm_freq = self.config.rm_optimization_frequency
        
        def rm_data_gen(dt: datetime) -> RMOptimizationEvent:
            # Find flights departing in optimization horizons
            flights_to_optimize = []
            for horizon_days in self.config.optimization_horizons:
                target_date = dt.date() + timedelta(days=horizon_days)
                for fd in self.flight_dates.values():
                    if fd.departure_date == target_date:
                        flights_to_optimize.append(fd)
            
            return RMOptimizationEvent(
                flight_dates=flights_to_optimize,
                optimization_method=self.config.rm_method
            )

        if rm_freq == "daily":
            count = self.event_scheduler.schedule_daily(
                event_type=EventType.RM_OPTIMIZATION,
                data_generator=rm_data_gen,
                start_date=start_dt,
                end_date=end_dt,
                time_of_day=datetime.min.time().replace(hour=2),  # 2 AM
                priority=EventPriority.HIGH.value
            )
            self.logger.info(f"Scheduled {count} RM optimization events (Daily)")
        elif isinstance(rm_freq, int):
            # Schedule at intervals
            count = self.event_scheduler.schedule_at_intervals(
                event_type=EventType.RM_OPTIMIZATION,
                data_generator=rm_data_gen,
                start_time=start_dt,
                end_time=end_dt,
                interval_hours=float(rm_freq),
                priority=EventPriority.HIGH.value
            )
            self.logger.info(f"Scheduled {count} RM optimization events (Every {rm_freq} hours)")
        
        # Schedule snapshots
        if self.config.save_snapshots:
            def snapshot_data_gen(dt: datetime) -> SnapshotEvent:
                return SnapshotEvent(
                    snapshot_id=f"snapshot_{dt.date().isoformat()}",
                    include_detailed_state=self.config.snapshot_detailed
                )
            
            # Schedule weekly snapshots
            count = self.event_scheduler.schedule_at_intervals(
                event_type=EventType.SNAPSHOT,
                data_generator=snapshot_data_gen,
                start_time=start_dt,
                end_time=end_dt,
                interval_hours=24 * self.config.snapshot_frequency_days,
                priority=EventPriority.LOW.value
            )
            self.logger.info(f"Scheduled {count} snapshot events")
    
    def _handle_booking_request(self, event: Event) -> Optional[Booking]:
        """Handle a booking request event."""
        data: BookingRequestEvent = event.data
        request = data.request
        
        self.logger.debug(f"Processing booking request: {request}")
        
        # Step 1: Search for travel solutions
        solutions = self._search_travel_solutions(request)
        
        if not solutions:
            self.logger.info(f"REJECTED - No availability: {request.origin.code}->{request.destination.code} "
                           f"on {request.departure_date}, {request.party_size} pax, "
                           f"segment={request.customer.segment.value}")
            # Record rejected request
            if self.data_exporter:
                self.data_exporter.add_request(request, accepted=False, 
                                              reason="no_availability",
                                              solutions=solutions)
            return None
        
        # Enrich customer with personalization data
        request.customer = self.personalization_engine.enrich_customer(request.customer)
        
        # Step 2: Calculate fares
        self._calculate_fares(solutions, request)
        
        # Step 3: Check availability
        self._check_availability(solutions)
        
        # Personalize solutions (adjust prices, create bundles)
        solutions = self.personalization_engine.personalize_solutions(solutions, request.customer)
        
        # Step 4: Customer choice
        chosen_solution = self._customer_choice(solutions, request.customer)
        
        if chosen_solution is None:
            self.logger.info(f"REJECTED - Customer declined: {request.origin.code}->{request.destination.code} "
                           f"on {request.departure_date}, {request.party_size} pax, "
                           f"WTP=${request.customer.willingness_to_pay:.0f}, "
                           f"segment={request.customer.segment.value}")
            # Record rejected request
            if self.data_exporter:
                self.data_exporter.add_request(request, accepted=False, 
                                              reason="customer_declined",
                                              solutions=solutions)
            return None
        
        # Step 5: Make booking
        booking = self._make_booking(chosen_solution, request)
        
        if booking:
            self.results.total_bookings += 1
            self.results.total_revenue += booking.total_revenue
            self.results.total_seats_sold += booking.party_size
            self.bookings.append(booking)
            self.results.bookings.append(booking)
            
            # Log successful booking
            flight = booking.solution.flights[0]
            booking_class = booking.solution.booking_classes[0]
            dtd = (flight.departure_date - booking.booking_time.date()).days
            self.logger.info(f"BOOKED - {booking.booking_id[:8]}: "
                           f"{flight.schedule.flight_code} {flight.departure_date} "
                           f"{flight.schedule.route.origin.code}->{flight.schedule.route.destination.code}, "
                           f"{booking.party_size} pax in {booking_class}, "
                           f"${booking.total_revenue:.2f}, "
                           f"DTD={dtd}, "
                           f"segment={booking.customer.segment.value}")
            
            # Record successful booking
            if self.data_exporter:
                self.data_exporter.add_booking(booking)
                self.data_exporter.add_request(request, accepted=True,
                                              solutions=solutions,
                                              chosen_solution=chosen_solution)
            
            # DB Insert
            if self.db:
                self.db.insert_customer(request.customer, request.request_id)
                self.db.insert_booking(booking)

            # Possibly generate cancellation event
            self._maybe_generate_cancellation(booking)
        else:
            self.logger.info(f"REJECTED - Booking failed: {request.origin.code}->{request.destination.code}")
            # Record rejected request
            if self.data_exporter:
                self.data_exporter.add_request(request, accepted=False, 
                                              reason="booking_failed",
                                              solutions=solutions,
                                              chosen_solution=chosen_solution)
            
            # DB Insert (Customer only for rejected)
            if self.db:
                self.db.insert_customer(request.customer, request.request_id)
        
        return booking
    
    def _handle_cancellation(self, event: Event) -> None:
        """Handle a cancellation event."""
        data: CancellationEvent = event.data
        booking = data.booking
        
        if booking.is_cancelled:
            return  # Already cancelled
            
        # Use Reservation System to cancel if possible
        if booking.record_locator:
            success = self.reservation_system.cancel_booking(booking.record_locator)
            if not success:
                self.logger.warning(f"Failed to cancel PNR {booking.record_locator} in PSS")
        else:
            # Fallback for legacy/direct bookings without PNR
            booking.is_cancelled = True
            booking.cancellation_time = event.timestamp
            
            # Restore inventory manually
            for flight, booking_class in zip(booking.solution.flights, booking.solution.booking_classes):
                current_bookings = flight.bookings.get(booking_class, 0)
                flight.bookings[booking_class] = max(0, current_bookings - booking.party_size)
        
        flight = booking.solution.flights[0]
        self.logger.info(f"CANCELLED - {booking.booking_id[:8]}: "
                       f"{flight.schedule.flight_code} {flight.departure_date}, "
                       f"{booking.party_size} pax, "
                       f"-${booking.total_revenue:.2f}")
        
        self.results.total_cancellations += 1
        self.results.total_seats_sold -= booking.party_size
        self.results.total_revenue -= booking.total_revenue
        
        # DB Update
        if self.db:
            self.db.insert_booking(booking)
    
    def _handle_rm_optimization(self, event: Event) -> None:
        """Handle RM optimization event."""
        data: RMOptimizationEvent = event.data
        
        self.logger.info(f"Running RM optimization for {len(data.flight_dates)} flights")
        
        for flight_date in data.flight_dates:
            try:
                # 1. Get Forecast
                forecast_obj = self.forecaster.forecast_demand(
                    flight_date, 
                    self.current_date, 
                    flight_date.bookings
                )
                
                # 2. Convert to OptimizerForecast
                optimizer_forecasts = []
                for bc, mean in forecast_obj.forecasts.items():
                    # Use Poisson assumption: std = sqrt(mean)
                    std = np.sqrt(mean) if mean > 0 else 0
                    
                    opt_forecast = OptimizerForecast(
                        booking_class=bc,
                        mean=mean,
                        std=std,
                        distribution="poisson"
                    )
                    optimizer_forecasts.append(opt_forecast)
                
                # 3. Get Fares
                fares = self._get_fares_for_optimization(flight_date)
                
                # 4. Optimize
                control = self.rm_optimizer.optimize(
                    flight_date,
                    optimizer_forecasts,
                    fares
                )
                
                # 5. Apply controls
                flight_date.booking_limits = control.booking_limits
                flight_date.bid_prices = control.bid_prices
                
                self.logger.debug(f"Optimized {flight_date.flight_id}: {control.booking_limits}")
                
            except Exception as e:
                self.logger.error(f"Error optimizing flight {flight_date.flight_id}: {str(e)}")
        
        self.results.rm_optimization_count += 1

    def _get_fares_for_optimization(self, flight_date: FlightDate) -> Dict[BookingClass, float]:
        """Get fares for optimization with realistic fare ladder."""
        fares = {}
        # Base fare calculation: $50 fixed + $0.08 per km (approx $0.13 per mile)
        # This represents a standard Economy (B class) fare
        base_fare = 50 + flight_date.schedule.route.distance_km * 0.08
        
        # Realistic fare multipliers relative to standard Economy (B class)
        multipliers = {
            # First Class
            BookingClass.F: 6.0,  # Full First
            BookingClass.A: 5.0,  # Discount First
            
            # Business Class
            BookingClass.J: 4.0,  # Full Business
            BookingClass.C: 3.5,
            BookingClass.D: 3.0,
            BookingClass.I: 2.5,  # Discount Business
            
            # Premium Economy
            BookingClass.W: 2.0,  # Full Premium Eco
            BookingClass.E: 1.7,  # Discount Premium Eco
            
            # Economy
            BookingClass.Y: 1.5,  # Full Fare Economy
            BookingClass.B: 1.0,  # Standard Economy (Reference)
            BookingClass.M: 0.9,
            BookingClass.H: 0.8,
            BookingClass.Q: 0.7,
            BookingClass.K: 0.6,
            BookingClass.L: 0.5   # Deep Discount Economy
        }
        
        for bc in BookingClass:
            multiplier = multipliers.get(bc, 1.0)
            fares[bc] = round(base_fare * multiplier, 2)
            
        return fares
    
    def _handle_price_update(self, event: Event) -> None:
        """Handle dynamic price update event."""
        # Placeholder for dynamic pricing
        pass
    
    def _handle_snapshot(self, event: Event) -> None:
        """Handle snapshot event."""
        data: SnapshotEvent = event.data
        self.logger.debug(f"Taking snapshot: {data.snapshot_id}")
        
        # Capture current state
        snapshot = {
            'timestamp': event.timestamp,
            'snapshot_id': data.snapshot_id,
            'current_date': self.current_date,
            'total_bookings': self.results.total_bookings,
            'total_revenue': self.results.total_revenue,
            'load_factor': self.results.load_factor,
        }
        
        # Store snapshot (would be more detailed in real implementation)
        # self.results.snapshots.append(snapshot)
    
    def _search_travel_solutions(self, request: BookingRequest) -> List[TravelSolution]:
        """
        Search for travel solutions matching request.
        
        Placeholder - actual implementation would use schedule search.
        """
        solutions = []
        
        # Find direct flights
        for flight_date in self.flight_dates.values():
            if (flight_date.schedule.route.origin == request.origin and
                flight_date.schedule.route.destination == request.destination and
                flight_date.departure_date == request.departure_date):
                
                # Create solution for each available booking class
                cabin = request.preferred_cabin
                for booking_class in BookingClass:
                    if booking_class.get_cabin() == cabin:
                        solution = TravelSolution(
                            flights=[flight_date],
                            booking_classes=[booking_class],
                            num_connections=0
                        )
                        solutions.append(solution)
        
        return solutions
    
    def _calculate_fares(self, solutions: List[TravelSolution], request: BookingRequest) -> None:
        """Calculate fares for solutions. Placeholder."""
        for solution in solutions:
            # Placeholder fare calculation
            booking_class = solution.booking_classes[0]
            
            # Simple distance-based pricing
            base_fare = 100 + solution.flights[0].schedule.route.distance_km * 0.10
            
            # Adjust by class
            if booking_class in [BookingClass.F, BookingClass.A]:
                base_fare *= 3.0
            elif booking_class in [BookingClass.J, BookingClass.C]:
                base_fare *= 2.0
            elif booking_class in [BookingClass.Y, BookingClass.B]:
                base_fare *= 1.0
            else:
                base_fare *= 0.7
            
            solution.base_fare = base_fare
            solution.taxes_fees = base_fare * 0.15
            solution.total_price = solution.base_fare + solution.taxes_fees
            
            self.logger.debug(f"Fare calculated for {booking_class}: ${solution.total_price:.2f}")
    
    def _check_availability(self, solutions: List[TravelSolution]) -> None:
        """Check and update availability for solutions."""
        for solution in solutions:
            # Check availability for first flight (simplified)
            flight = solution.flights[0]
            booking_class = solution.booking_classes[0]
            solution.available_seats = flight.available_seats(booking_class)
    
    def _customer_choice(self, solutions: List[TravelSolution], customer: Customer) -> Optional[TravelSolution]:
        """
        Simulate customer choice from available solutions.
        
        Uses FRAT5 logic for sell-up if the preferred option is not available.
        """
        if not solutions:
            return None
            
        # Use configured choice model
        if self.choice_model is not None:
            from choice.models import ChoiceSet
            choice_set = ChoiceSet(
                own_solutions=solutions,
                competitor_solutions=[],
                no_purchase_utility=0.0
            )
            
            # Use numpy RNG seeded from config
            rng = np.random.default_rng(self.config.random_seed)
            
            if hasattr(self.choice_model, 'predict_choice_with_behavior'):
                # Enhanced model with buy-up/down
                return self.choice_model.predict_choice_with_behavior(
                    choice_set, customer, None, rng
                )
            else:
                # Standard MNL
                return self.choice_model.predict_choice(choice_set, customer, rng)
        
        # Fallback: Simple logic with FRAT5 Sell-up
        # 1. Identify available solutions
        available = [s for s in solutions if s.available_seats > 0]
        
        if not available:
            return None
            
        # 2. Identify the "Lowest Logical Fare" (LLF) - the reference price
        # This is the cheapest fare that exists for this route, even if closed.
        # In this simplified simulation, we assume the cheapest solution in the full list 
        # (including closed ones) is the reference.
        all_prices = [s.total_price for s in solutions]
        lowest_logical_fare = min(all_prices) if all_prices else 0
        
        # 3. Filter by WTP first (hard constraint)
        affordable = [s for s in available if s.total_price <= customer.willingness_to_pay]
        
        if not affordable:
            # If nothing is strictly within WTP, check if they might stretch for a sell-up
            # using FRAT5 logic against the lowest logical fare.
            # This allows "soft" WTP limits.
            
            # Sort by price
            available.sort(key=lambda s: s.total_price)
            cheapest_available = available[0]
            
            # Calculate sell-up probability from LLF to this available fare
            prob = self.frat5_model.calculate_sellup_prob(
                lower_fare=lowest_logical_fare,
                higher_fare=cheapest_available.total_price,
                segment=customer.segment
            )
            
            # Random draw
            import random
            if random.random() < prob:
                return cheapest_available
            
            return None
        
        # 4. If there are affordable options, pick the cheapest one
        # (Standard behavior for price-sensitive customers)
        return min(affordable, key=lambda s: s.total_price)
    
    def _make_booking(self, solution: TravelSolution, request: BookingRequest) -> Optional[Booking]:
        """Create booking and update inventory using Reservation System."""
        # Simplified: We only book the first segment through the PSS for now
        # In a real system, we would book all segments into one PNR
        
        if not solution.flights:
            return None
            
        flight = solution.flights[0]
        booking_class = solution.booking_classes[0]
        price_per_pax = solution.total_price / request.party_size if request.party_size > 0 else 0
        
        # Use Reservation System to create PNR and update inventory
        pnr = self.reservation_system.create_booking(
            flight_date=flight,
            booking_class=booking_class,
            customer=request.customer,
            party_size=request.party_size,
            price=price_per_pax,
            channel=request.channel,
            booking_time=request.request_time
        )
        
        if pnr:
            booking = pnr.bookings[0]
            booking.original_request = request
            # Ensure the booking has the full solution details
            booking.solution = solution
            
            # Set currency info
            booking.currency = self.config.customer_currency
            booking.exchange_rate = self.config.currency_rate
            
            return booking
            
        return None
    
    def _maybe_generate_cancellation(self, booking: Booking) -> None:
        """Probabilistically generate a cancellation event."""
        import random
        
        # Cancellation rate varies by segment
        cancel_prob = 0.05  # 5% base rate
        
        if booking.customer.segment == CustomerSegment.BUSINESS:
            cancel_prob = 0.10  # Business travelers cancel more
        
        if random.random() < cancel_prob:
            # Schedule cancellation 1-30 days after booking
            days_until = random.randint(1, 30)
            cancel_time = booking.booking_time + timedelta(days=days_until)
            
            # Don't cancel after departure
            departure = booking.solution.departure_time
            if cancel_time < departure:
                self.event_manager.create_and_add_event(
                    timestamp=cancel_time,
                    event_type=EventType.CANCELLATION,
                    data=CancellationEvent(booking=booking)
                )
    
    def run(self) -> SimulationResults:
        """
        Run the complete simulation.
        
        Returns:
            SimulationResults with all metrics and data
        """
        self.logger.info("="*60)
        self.logger.info("Starting Simulation")
        self.logger.info("="*60)
        self.logger.info(f"Period: {self.config.start_date} to {self.config.end_date}")
        self.logger.info(f"Flights: {len(self.flight_dates)}")
        self.logger.info(f"Events Queued: {self.event_manager.size()}")
        
        self.status = SimulationStatus.RUNNING
        self.results.start_time = datetime.now()
        
        # Setup progress bar
        total_events = self.event_manager.size()
        pbar = None
        if self.config.progress_bar:
            pbar = tqdm(total=total_events, desc="Processing Events")
        
        try:
            # Main event loop
            while not self.event_manager.is_empty():
                event = self.event_manager.pop_next_event()
                
                # Update current date
                if event.timestamp.date() != self.current_date:
                    self.current_date = event.timestamp.date()
                    self.logger.debug(f"Simulation date: {self.current_date}")
                
                # Process event
                self.event_manager.process_event(event)
                
                if pbar:
                    pbar.update(1)
                
                # Debug logging
                if self.event_manager.size() % 100 == 0:
                    print(f"DEBUG: Remaining events: {self.event_manager.size()}", flush=True)
            
            self.status = SimulationStatus.COMPLETED
            
        except Exception as e:
            self.logger.error(f"Simulation error: {e}", exc_info=True)
            self.status = SimulationStatus.ERROR
            raise
        
        finally:
            if pbar:
                pbar.close()
            
            self.results.end_time = datetime.now()
            self.results.flights_operated = len(self.flight_dates)
            self.results.total_seats_offered = sum(
                sum(fd.capacity.values()) for fd in self.flight_dates.values()
            )
        
        self.logger.info("="*60)
        self.logger.info("Simulation Complete")
        self.logger.info("="*60)
        self.logger.info(self.results.summary())
        
        # Export data to CSV files
        if self.data_exporter:
            self.logger.info("Exporting simulation data to CSV files...")
            exported_files = self.data_exporter.export_all()
            self.results.exported_files = {k: str(v) for k, v in exported_files.items()}
            for data_type, filepath in exported_files.items():
                self.logger.info(f"  - {data_type}: {filepath}")
            self.logger.info(f"CSV exports complete: {len(exported_files)} files created")
            
        # Update DB stats
        if self.db:
            self.logger.info("Updating database statistics...")
            for fd in self.flight_dates.values():
                self.db.update_flight_stats(fd)
            self.db.update_flight_revenue_from_bookings()
            self.logger.info("Database update complete.")
        
        return self.results
    
    def reset(self) -> None:
        """Reset simulator to initial state."""
        self.event_manager.clear_queue()
        self.flight_dates.clear()
        self.bookings.clear()
        self.results = SimulationResults(config=self.config)
        self.status = SimulationStatus.INITIALIZED
        self._initialize_components()
