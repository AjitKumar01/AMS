"""
Reservation System (PSS) Integration.

This module simulates a Passenger Service System (PSS) or Reservation System.
It handles the transactional aspect of making bookings, creating PNRs,
and managing inventory updates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import uuid
import random
import string

from core.models import Booking, FlightDate, BookingClass, Customer, BookingChannel, TravelSolution

@dataclass
class PNR:
    """
    Passenger Name Record (PNR).
    Represents a confirmed reservation in the system.
    """
    record_locator: str
    bookings: List[Booking]
    customer: Customer
    creation_date: datetime
    status: str = "CONFIRMED"
    ticketing_time_limit: Optional[datetime] = None
    special_requests: List[str] = field(default_factory=list)

class ReservationSystem:
    """
    Simulated Reservation System (PSS).
    
    Responsibilities:
    1. Inventory Availability Check
    2. Booking Creation (PNR generation)
    3. Ticket Issuance
    4. Cancellation & Modification
    """
    
    def __init__(self):
        self.pnrs: Dict[str, PNR] = {}
        
    def check_availability(
        self, 
        flight_date: FlightDate, 
        booking_class: BookingClass, 
        party_size: int
    ) -> bool:
        """
        Check if requested seats are available.
        
        Args:
            flight_date: The flight to check
            booking_class: The booking class (fare bucket)
            party_size: Number of seats requested
            
        Returns:
            True if available, False otherwise
        """
        available_seats = flight_date.available_seats(booking_class)
        return available_seats >= party_size

    def create_booking(
        self,
        flight_date: FlightDate,
        booking_class: BookingClass,
        customer: Customer,
        party_size: int,
        price: float,
        channel: BookingChannel = BookingChannel.DIRECT_ONLINE,
        booking_time: Optional[datetime] = None
    ) -> Optional[PNR]:
        """
        Create a new booking and PNR.
        
        Args:
            flight_date: The flight to book
            booking_class: The booking class
            customer: The customer making the booking
            party_size: Number of passengers
            price: Price per passenger
            channel: Booking channel
            booking_time: Time of booking (defaults to now)
            
        Returns:
            PNR object if successful, None if failed (no availability)
        """
        # 1. Check availability
        if not self.check_availability(flight_date, booking_class, party_size):
            return None
            
        # 2. Create TravelSolution (Single leg for now)
        solution = TravelSolution(
            flights=[flight_date],
            booking_classes=[booking_class],
            total_price=price * party_size,
            available_seats=party_size
        )
            
        # 3. Create Booking object
        timestamp = booking_time or datetime.now()
        booking = Booking(
            booking_id=str(uuid.uuid4()),
            booking_time=timestamp,
            customer=customer,
            solution=solution,
            party_size=party_size,
            total_revenue=price * party_size
        )
        
        # 4. Update Inventory (Decrement availability)
        # In a real system, this would be a transaction.
        # Here we update the FlightDate object directly.
        current_bookings = flight_date.bookings.get(booking_class, 0)
        flight_date.bookings[booking_class] = current_bookings + party_size
        
        # 5. Generate PNR
        record_locator = self._generate_record_locator()
        
        # Link booking to PNR
        booking.record_locator = record_locator
        
        pnr = PNR(
            record_locator=record_locator,
            bookings=[booking],
            customer=customer,
            creation_date=timestamp
        )
        
        # 6. Store PNR
        self.pnrs[record_locator] = pnr
        
        return pnr

    def cancel_booking(self, record_locator: str) -> bool:
        """
        Cancel a booking by PNR.
        
        Args:
            record_locator: The PNR record locator
            
        Returns:
            True if cancelled, False if not found
        """
        if record_locator not in self.pnrs:
            return False
            
        pnr = self.pnrs[record_locator]
        pnr.status = "CANCELLED"
        
        # Release inventory
        for booking in pnr.bookings:
            booking.is_cancelled = True
            booking.cancellation_time = datetime.now()
            
            # Iterate through segments in the solution
            if booking.solution:
                for i, flight_date in enumerate(booking.solution.flights):
                    if i < len(booking.solution.booking_classes):
                        bc = booking.solution.booking_classes[i]
                        
                        current_bookings = flight_date.bookings.get(bc, 0)
                        # Ensure we don't go below zero
                        new_bookings = max(0, current_bookings - booking.party_size)
                        flight_date.bookings[bc] = new_bookings
            
        return True

    def retrieve_pnr(self, record_locator: str) -> Optional[PNR]:
        """Retrieve a PNR by record locator."""
        return self.pnrs.get(record_locator)

    def _generate_record_locator(self) -> str:
        """Generate a unique 6-character alphanumeric PNR."""
        while True:
            # Standard airline PNR format (6 chars, alphanumeric)
            chars = string.ascii_uppercase + string.digits
            locator = ''.join(random.choices(chars, k=6))
            if locator not in self.pnrs:
                return locator
