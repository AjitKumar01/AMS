"""
Event management system for discrete event simulation.

This module implements a priority queue-based event system for
chronological processing of booking requests, cancellations,
RM optimizations, and other time-based events.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional, List, Callable, Dict
from queue import PriorityQueue
from enum import Enum
import heapq
from core.models import EventType, BookingRequest, Booking, FlightDate


@dataclass(order=True)
class Event:
    """Simulation event with automatic priority ordering."""
    
    # Priority fields (used for sorting)
    timestamp: datetime = field(compare=True)
    event_type: EventType = field(compare=False)  # Required field (no default)
    priority: int = field(default=0, compare=True)  # Lower = higher priority
    
    # Event data (not used for comparison)
    data: Any = field(default=None, compare=False)
    event_id: str = field(default="", compare=False)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now, compare=False)
    processed: bool = field(default=False, compare=False)
    processing_time_ms: float = field(default=0.0, compare=False)
    
    def __str__(self) -> str:
        return f"Event({self.event_type.value} at {self.timestamp})"
    
    def __hash__(self) -> int:
        return hash(self.event_id)


class EventPriority(Enum):
    """Priority levels for event processing."""
    CRITICAL = 0  # System events
    HIGH = 1  # RM optimization, price updates
    NORMAL = 2  # Booking requests
    LOW = 3  # Snapshots, reporting


@dataclass
class BookingRequestEvent:
    """Data for booking request events."""
    request: BookingRequest
    demand_stream_id: str  # Which demand stream generated this


@dataclass
class CancellationEvent:
    """Data for cancellation events."""
    booking: Booking
    reason: str = "customer_initiated"


@dataclass
class NoShowEvent:
    """Data for no-show events."""
    booking: Booking


@dataclass
class RMOptimizationEvent:
    """Data for RM optimization events."""
    flight_dates: List[FlightDate]
    optimization_method: str = "EMSR-b"
    trigger: str = "scheduled"  # 'scheduled' or 'demand_triggered'


@dataclass
class PriceUpdateEvent:
    """Data for dynamic pricing update events."""
    flight_dates: List[FlightDate]
    trigger: str = "scheduled"


@dataclass
class SnapshotEvent:
    """Data for state snapshot events."""
    snapshot_id: str
    include_detailed_state: bool = False


@dataclass
class CompetitorActionEvent:
    """Data for competitor actions."""
    competitor_airline: str
    action_type: str  # 'price_change', 'capacity_change', 'schedule_change'
    details: Dict[str, Any] = field(default_factory=dict)


class EventManager:
    """
    Manages simulation events with priority queue.
    
    Features:
    - Priority-based event processing
    - Event type filtering
    - Progress tracking
    - Event history
    """
    
    def __init__(self):
        self._event_queue: List[Event] = []
        self._event_history: List[Event] = []
        self._event_handlers: Dict[EventType, List[Callable]] = {}
        self._current_time: datetime = datetime.now()
        
        # Statistics
        self._events_processed = 0
        self._events_by_type: Dict[EventType, int] = {et: 0 for et in EventType}
        
    def add_event(self, event: Event) -> None:
        """Add event to the queue."""
        heapq.heappush(self._event_queue, event)
    
    def create_and_add_event(
        self,
        timestamp: datetime,
        event_type: EventType,
        data: Any,
        priority: int = EventPriority.NORMAL.value,
        event_id: Optional[str] = None
    ) -> Event:
        """Create and add event in one step."""
        if event_id is None:
            event_id = f"{event_type.value}_{timestamp.isoformat()}_{self._events_processed}"
        
        event = Event(
            timestamp=timestamp,
            priority=priority,
            event_type=event_type,
            data=data,
            event_id=event_id
        )
        self.add_event(event)
        return event
    
    def pop_next_event(self) -> Optional[Event]:
        """Get and remove next event from queue."""
        if self.is_empty():
            return None
        
        event = heapq.heappop(self._event_queue)
        self._current_time = event.timestamp
        return event
    
    def peek_next_event(self) -> Optional[Event]:
        """Look at next event without removing it."""
        if self.is_empty():
            return None
        return self._event_queue[0]
    
    def is_empty(self) -> bool:
        """Check if event queue is empty."""
        return len(self._event_queue) == 0
    
    def size(self) -> int:
        """Get number of events in queue."""
        return len(self._event_queue)
    
    def register_handler(self, event_type: EventType, handler: Callable) -> None:
        """
        Register an event handler function.
        
        Handler signature: handler(event: Event) -> Any
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def process_event(self, event: Event) -> Any:
        """Process an event by calling registered handlers."""
        if event.processed:
            return None
        
        start_time = datetime.now()
        results = []
        
        # Call all registered handlers for this event type
        handlers = self._event_handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                result = handler(event)
                results.append(result)
            except Exception as e:
                print(f"Error processing event {event}: {e}")
                raise
        
        # Mark as processed
        event.processed = True
        event.processing_time_ms = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update statistics
        self._events_processed += 1
        self._events_by_type[event.event_type] += 1
        
        # Store in history
        self._event_history.append(event)
        
        return results[0] if len(results) == 1 else results
    
    def process_next_event(self) -> Optional[Any]:
        """Pop and process next event."""
        event = self.pop_next_event()
        if event is None:
            return None
        return self.process_event(event)
    
    def process_all_events(self, max_events: Optional[int] = None) -> int:
        """Process all events in queue (or up to max_events)."""
        count = 0
        while not self.is_empty():
            if max_events is not None and count >= max_events:
                break
            self.process_next_event()
            count += 1
        return count
    
    def process_until(self, end_time: datetime) -> int:
        """Process events until specified time."""
        count = 0
        while not self.is_empty():
            next_event = self.peek_next_event()
            if next_event.timestamp > end_time:
                break
            self.process_next_event()
            count += 1
        return count
    
    def get_events_by_type(self, event_type: EventType) -> List[Event]:
        """Get all events of a specific type currently in queue."""
        return [e for e in self._event_queue if e.event_type == event_type]
    
    def get_events_in_timerange(
        self, 
        start_time: datetime, 
        end_time: datetime
    ) -> List[Event]:
        """Get events within time range."""
        return [e for e in self._event_queue 
                if start_time <= e.timestamp <= end_time]
    
    def clear_queue(self) -> None:
        """Clear all events from queue."""
        self._event_queue.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event processing statistics."""
        return {
            'events_processed': self._events_processed,
            'events_queued': len(self._event_queue),
            'events_by_type': dict(self._events_by_type),
            'current_time': self._current_time,
            'queue_size': self.size(),
            'history_size': len(self._event_history)
        }
    
    @property
    def current_time(self) -> datetime:
        """Get current simulation time."""
        return self._current_time
    
    def __len__(self) -> int:
        return len(self._event_queue)
    
    def __str__(self) -> str:
        return f"EventManager({len(self._event_queue)} events queued, {self._events_processed} processed)"


class EventScheduler:
    """
    Helper class for scheduling recurring events.
    
    Useful for scheduling regular RM optimizations, snapshots, etc.
    """
    
    def __init__(self, event_manager: EventManager):
        self.event_manager = event_manager
    
    def schedule_daily(
        self,
        event_type: EventType,
        data_generator: Callable[[datetime], Any],
        start_date: datetime,
        end_date: datetime,
        time_of_day: datetime.time = datetime.min.time(),
        priority: int = EventPriority.HIGH.value
    ) -> int:
        """
        Schedule daily recurring events.
        
        Args:
            event_type: Type of event to schedule
            data_generator: Function that generates event data given timestamp
            start_date: First occurrence
            end_date: Last occurrence
            time_of_day: Time of day to schedule
            priority: Event priority
            
        Returns:
            Number of events scheduled
        """
        count = 0
        current = start_date.replace(hour=time_of_day.hour, minute=time_of_day.minute)
        
        while current <= end_date:
            data = data_generator(current)
            self.event_manager.create_and_add_event(
                timestamp=current,
                event_type=event_type,
                data=data,
                priority=priority
            )
            current += timedelta(days=1)
            count += 1
        
        return count
    
    def schedule_at_intervals(
        self,
        event_type: EventType,
        data_generator: Callable[[datetime], Any],
        start_time: datetime,
        end_time: datetime,
        interval_hours: float,
        priority: int = EventPriority.NORMAL.value
    ) -> int:
        """
        Schedule events at regular intervals.
        
        Args:
            event_type: Type of event to schedule
            data_generator: Function that generates event data
            start_time: First event time
            end_time: Last possible event time
            interval_hours: Hours between events
            priority: Event priority
            
        Returns:
            Number of events scheduled
        """
        from datetime import timedelta
        
        count = 0
        current = start_time
        interval = timedelta(hours=interval_hours)
        
        while current <= end_time:
            data = data_generator(current)
            self.event_manager.create_and_add_event(
                timestamp=current,
                event_type=event_type,
                data=data,
                priority=priority
            )
            current += interval
            count += 1
        
        return count
    
    def schedule_before_departures(
        self,
        event_type: EventType,
        flight_dates: List[FlightDate],
        hours_before: float,
        data_generator: Callable[[FlightDate], Any],
        priority: int = EventPriority.HIGH.value
    ) -> int:
        """
        Schedule events before flight departures.
        
        Useful for RM optimizations at specific DTD thresholds.
        
        Args:
            event_type: Type of event to schedule
            flight_dates: Flights to schedule events for
            hours_before: Hours before departure to schedule
            data_generator: Function that generates event data
            priority: Event priority
            
        Returns:
            Number of events scheduled
        """
        from datetime import timedelta
        
        count = 0
        for flight_date in flight_dates:
            event_time = flight_date.departure_datetime - timedelta(hours=hours_before)
            data = data_generator(flight_date)
            
            self.event_manager.create_and_add_event(
                timestamp=event_time,
                event_type=event_type,
                data=data,
                priority=priority
            )
            count += 1
        
        return count


class ProgressTracker:
    """Track simulation progress."""
    
    def __init__(self, total_events: int):
        self.total_events = total_events
        self.events_processed = 0
        self.start_time = datetime.now()
        
    def update(self, events_processed: int) -> None:
        """Update progress."""
        self.events_processed = events_processed
    
    def get_progress(self) -> float:
        """Get progress as percentage (0.0 to 1.0)."""
        if self.total_events == 0:
            return 0.0
        return min(1.0, self.events_processed / self.total_events)
    
    def get_eta(self) -> Optional[datetime]:
        """Estimate time to completion."""
        if self.events_processed == 0:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.events_processed / elapsed
        remaining = self.total_events - self.events_processed
        
        if rate <= 0:
            return None
        
        seconds_remaining = remaining / rate
        from datetime import timedelta
        return datetime.now() + timedelta(seconds=seconds_remaining)
    
    def __str__(self) -> str:
        pct = self.get_progress() * 100
        return f"Progress: {pct:.1f}% ({self.events_processed}/{self.total_events})"
