
import sqlite3
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import json
from pathlib import Path

from core.models import Booking, Customer, FlightDate, BookingRequest

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Manages SQLite database for simulation results.
    """
    
    def __init__(self, db_path: str = "simulation_results/simulation.db"):
        self.db_path = db_path
        self._create_connection()
        self.create_tables()
        
    def _create_connection(self):
        """Create a database connection."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        
    def create_tables(self):
        """Create the necessary tables."""
        cursor = self.conn.cursor()
        
        # Customers Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            request_id TEXT,
            segment TEXT,
            willingness_to_pay REAL,
            price_sensitivity REAL,
            time_sensitivity REAL,
            loyalty_tier TEXT,
            ancillary_prefs TEXT
        )
        """)
        
        # Bookings Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id TEXT PRIMARY KEY,
            request_id TEXT,
            customer_id TEXT,
            booking_time TEXT,
            flight_code TEXT,
            departure_date TEXT,
            origin TEXT,
            destination TEXT,
            cabin_class TEXT,
            booking_class TEXT,
            base_fare REAL,
            total_price REAL,
            ancillaries TEXT,
            is_personalized BOOLEAN,
            is_cancelled BOOLEAN,
            FOREIGN KEY(customer_id) REFERENCES customers(customer_id)
        )
        """)
        
        # Flights Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS flights (
            flight_id TEXT PRIMARY KEY,
            flight_code TEXT,
            departure_date TEXT,
            origin TEXT,
            destination TEXT,
            capacity INTEGER,
            bookings_count INTEGER,
            revenue REAL,
            load_factor REAL,
            avg_fare REAL
        )
        """)
        
        self.conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

    def insert_customer(self, customer: Customer, request_id: str):
        """Insert a customer record."""
        cursor = self.conn.cursor()
        
        loyalty_tier = ''
        if hasattr(customer, 'loyalty_tier') and customer.loyalty_tier:
            loyalty_tier = customer.loyalty_tier.value if hasattr(customer.loyalty_tier, 'value') else str(customer.loyalty_tier)
        
        ancillary_prefs = ''
        if hasattr(customer, 'ancillary_preferences') and customer.ancillary_preferences:
            ancillary_prefs = "|".join(customer.ancillary_preferences)
            
        cursor.execute("""
        INSERT OR REPLACE INTO customers (
            customer_id, request_id, segment, willingness_to_pay, 
            price_sensitivity, time_sensitivity, loyalty_tier, ancillary_prefs
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            customer.customer_id,
            request_id,
            customer.segment.value,
            customer.willingness_to_pay,
            customer.price_sensitivity,
            customer.time_sensitivity,
            loyalty_tier,
            ancillary_prefs
        ))
        self.conn.commit()

    def insert_booking(self, booking: Booking):
        """Insert a booking record."""
        cursor = self.conn.cursor()
        
        # Extract details
        flight = booking.solution.flights[0] if booking.solution and booking.solution.flights else None
        flight_code = flight.schedule.flight_code if flight else ""
        departure_date = flight.departure_date.isoformat() if flight else ""
        
        origin = booking.solution.origin.code if booking.solution else ""
        destination = booking.solution.destination.code if booking.solution else ""
        
        cabin = booking.solution.booking_classes[0].get_cabin().value if booking.solution else ""
        b_class = booking.solution.booking_classes[0].value if booking.solution else ""
        
        ancillaries = ""
        if booking.solution and booking.solution.ancillaries:
            ancillaries = "|".join(booking.solution.ancillaries)
            
        is_personalized = False
        if booking.solution:
            is_personalized = booking.solution.is_personalized_offer
            
        request_id = booking.original_request.request_id if booking.original_request else ""
        
        cursor.execute("""
        INSERT OR REPLACE INTO bookings (
            booking_id, request_id, customer_id, booking_time, 
            flight_code, departure_date, origin, destination,
            cabin_class, booking_class, base_fare, total_price,
            ancillaries, is_personalized, is_cancelled
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            booking.booking_id,
            request_id,
            booking.customer.customer_id,
            booking.booking_time.isoformat(),
            flight_code,
            departure_date,
            origin,
            destination,
            cabin,
            b_class,
            booking.solution.base_fare if booking.solution else 0.0,
            booking.total_revenue,
            ancillaries,
            is_personalized,
            booking.is_cancelled
        ))
        self.conn.commit()

    def update_flight_stats(self, flight: FlightDate):
        """Update flight statistics."""
        cursor = self.conn.cursor()
        
        revenue = sum(b.total_paid for b in flight.bookings.values() if hasattr(b, 'total_paid')) 
        # Note: FlightDate.bookings is Dict[BookingClass, int], not list of Booking objects.
        # We need to calculate revenue differently or pass it in.
        # Actually, FlightDate doesn't store revenue directly usually.
        # But let's use what we have.
        
        # For now, let's just store capacity and counts.
        # Revenue calculation might need to be done at the end or aggregated from bookings table.
        
        cursor.execute("""
        INSERT OR REPLACE INTO flights (
            flight_id, flight_code, departure_date, origin, destination,
            capacity, bookings_count, revenue, load_factor, avg_fare
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            flight.flight_id,
            flight.schedule.flight_code,
            flight.departure_date.isoformat(),
            flight.schedule.route.origin.code,
            flight.schedule.route.destination.code,
            sum(flight.capacity.values()),
            flight.total_bookings(),
            0.0, # Placeholder, will update later via SQL aggregation
            flight.load_factor(),
            0.0  # Placeholder
        ))
        self.conn.commit()
        
    def update_flight_revenue_from_bookings(self):
        """Recalculate flight revenue from bookings table."""
        cursor = self.conn.cursor()
        cursor.execute("""
        UPDATE flights 
        SET revenue = (
            SELECT SUM(total_price) 
            FROM bookings 
            WHERE bookings.flight_code = flights.flight_code 
            AND bookings.departure_date = flights.departure_date
            AND bookings.is_cancelled = 0
        ),
        bookings_count = (
             SELECT COUNT(*) 
            FROM bookings 
            WHERE bookings.flight_code = flights.flight_code 
            AND bookings.departure_date = flights.departure_date
            AND bookings.is_cancelled = 0
        )
        """)
        
        # Update avg_fare
        cursor.execute("""
        UPDATE flights
        SET avg_fare = CASE WHEN bookings_count > 0 THEN revenue / bookings_count ELSE 0 END
        """)
        
        self.conn.commit()

    def get_table_data(self, table_name: str) -> List[Dict[str, Any]]:
        """Fetch all data from a table."""
        cursor = self.conn.cursor()
        try:
            cursor.execute(f"SELECT * FROM {table_name}")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Error fetching from {table_name}: {e}")
            return []
