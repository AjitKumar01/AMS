"""
Data export module for simulation results.

This module provides comprehensive CSV export functionality for:
- Booking requests by days-to-departure
- Flight-level metrics
- Revenue and load factor analysis
- Customer segment analysis
- Overbooking performance
- Customer choice behavior
"""

import csv
from datetime import datetime, date
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from core.models import (
    Booking, BookingRequest, FlightDate, Customer, CustomerSegment,
    CabinClass, BookingClass
)

logger = logging.getLogger(__name__)


class DataExporter:
    """
    Export simulation data to CSV files.
    
    Creates structured CSV files for analysis:
    - bookings.csv: All booking details
    - requests.csv: All booking requests (accepted and rejected)
    - flights.csv: Flight-level performance metrics
    - dtd_analysis.csv: Days-to-departure booking patterns
    - segment_analysis.csv: Customer segment performance
    - overbooking.csv: Overbooking statistics per flight
    - revenue_timeline.csv: Revenue accumulation over time
    """
    
    def __init__(self, output_dir: str = "simulation_output"):
        """
        Initialize data exporter.
        
        Args:
            output_dir: Directory to save CSV files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data collectors
        self.bookings: List[Booking] = []
        self.requests: List[BookingRequest] = []
        self.detailed_requests: List[Dict[str, Any]] = []
        self.flight_snapshots: List[Dict[str, Any]] = []
        self.revenue_timeline: List[Dict[str, Any]] = []
        
        logger.info(f"Data exporter initialized. Output directory: {self.output_dir}")
    
    def add_booking(self, booking: Booking) -> None:
        """Record a booking."""
        self.bookings.append(booking)
    
    def add_request(self, request: BookingRequest, accepted: bool, reason: str = "",
                   solutions: List[Any] = None, chosen_solution: Any = None) -> None:
        """Record a booking request."""
        request.accepted = accepted
        request.rejection_reason = reason if not accepted else ""
        self.requests.append(request)
        
        # Store detailed info
        self.detailed_requests.append({
            'request': request,
            'accepted': accepted,
            'reason': reason,
            'solutions': solutions,
            'chosen_solution': chosen_solution
        })
    
    def add_flight_snapshot(self, flight: FlightDate, timestamp: datetime) -> None:
        """Record flight state at a point in time."""
        snapshot = {
            'timestamp': timestamp,
            'flight_id': flight.flight_id,
            'departure_date': flight.departure_date,
            'flight_code': flight.schedule.flight_code,
            'days_to_departure': (flight.departure_date - timestamp.date()).days,
            'total_bookings': flight.total_bookings(),
            'capacity': flight.schedule.aircraft.total_capacity,
            'load_factor': flight.load_factor(),
            'revenue': sum(b.total_paid for b in self.bookings if b.flight_date == flight)
        }
        self.flight_snapshots.append(snapshot)
    
    def add_revenue_snapshot(self, timestamp: datetime, total_revenue: float, 
                            bookings_count: int) -> None:
        """Record cumulative revenue at a point in time."""
        self.revenue_timeline.append({
            'timestamp': timestamp,
            'total_revenue': total_revenue,
            'bookings_count': bookings_count
        })
    
    def export_bookings(self, filename: str = "bookings.csv") -> str:
        """
        Export all bookings to CSV.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'booking_id', 'request_id', 'booking_time', 'departure_date', 'flight_code',
                'origin', 'destination', 'cabin_class', 'booking_class',
                'customer_segment', 'party_size', 'base_fare', 'total_paid',
                'days_to_departure', 'willingness_to_pay', 'price_sensitivity',
                'cancelled', 'cancellation_time',
                'loyalty_tier', 'ancillaries', 'is_personalized'
            ])
            
            # Data
            for booking in self.bookings:
                # Get flight info from solution
                if booking.solution and booking.solution.flights:
                    flight = booking.solution.flights[0]
                    departure_date = flight.departure_date
                    flight_code = flight.schedule.flight_code
                    dtd = (departure_date - booking.booking_time.date()).days
                else:
                    departure_date = None
                    flight_code = ''
                    dtd = 0
                
                # Get origin/destination from solution or original request
                if booking.solution and booking.solution.flights:
                    origin = booking.solution.flights[0].schedule.route.origin
                    destination = booking.solution.flights[-1].schedule.route.destination
                elif booking.original_request:
                    origin = booking.original_request.origin
                    destination = booking.original_request.destination
                else:
                    origin = None
                    destination = None
                
                # Get cabin and booking class from solution
                cabin_class = booking.solution.booking_classes[0].get_cabin().value \
                    if (booking.solution and booking.solution.booking_classes) else ''
                booking_class = booking.solution.booking_classes[0].value \
                    if (booking.solution and booking.solution.booking_classes) else ''
                
                # Personalization data
                loyalty_tier = ''
                if hasattr(booking.customer, 'loyalty_tier') and booking.customer.loyalty_tier:
                    loyalty_tier = booking.customer.loyalty_tier.value if hasattr(booking.customer.loyalty_tier, 'value') else str(booking.customer.loyalty_tier)
                
                ancillaries = ''
                if hasattr(booking, 'ancillaries') and booking.ancillaries:
                    ancillaries = "|".join(booking.ancillaries)
                
                is_personalized = False
                if hasattr(booking, 'is_personalized_offer'):
                    is_personalized = booking.is_personalized_offer
                
                request_id = ''
                if booking.original_request:
                    request_id = booking.original_request.request_id

                writer.writerow([
                    booking.booking_id,
                    request_id,
                    booking.booking_time.isoformat() if booking.booking_time else '',
                    departure_date.isoformat() if departure_date else '',
                    flight_code,
                    origin.code if origin else '',
                    destination.code if destination else '',
                    cabin_class,
                    booking_class,
                    booking.customer.segment.value if booking.customer else '',
                    booking.party_size,
                    f"{booking.solution.total_price if booking.solution else 0:.2f}",
                    f"{booking.total_revenue:.2f}",
                    dtd,
                    f"{booking.customer.willingness_to_pay:.2f}" if booking.customer else '',
                    f"{booking.customer.price_sensitivity:.2f}" if booking.customer else '',
                    booking.is_cancelled,
                    booking.cancellation_time.isoformat() if booking.cancellation_time else '',
                    loyalty_tier,
                    ancillaries,
                    is_personalized
                ])
        
        logger.info(f"Exported {len(self.bookings)} bookings to {filepath}")
        return str(filepath)
    
    def export_requests(self, filename: str = "booking_requests.csv") -> str:
        """
        Export all booking requests (accepted and rejected) to CSV.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'request_id', 'request_time', 'departure_date', 
                'origin', 'destination', 'preferred_cabin', 'preferred_class',
                'customer_segment', 'party_size', 'willingness_to_pay',
                'price_sensitivity', 'time_sensitivity', 'days_to_departure',
                'accepted', 'rejection_reason',
                'loyalty_tier', 'ancillary_prefs'
            ])
            
            # Data
            for req in self.requests:
                dtd = (req.departure_date - req.request_time.date()).days
                
                # Personalization data
                loyalty_tier = ''
                if hasattr(req.customer, 'loyalty_tier') and req.customer.loyalty_tier:
                    loyalty_tier = req.customer.loyalty_tier.value if hasattr(req.customer.loyalty_tier, 'value') else str(req.customer.loyalty_tier)
                
                ancillary_prefs = ''
                if hasattr(req.customer, 'ancillary_preferences') and req.customer.ancillary_preferences:
                    ancillary_prefs = "|".join(req.customer.ancillary_preferences)

                writer.writerow([
                    req.request_id,
                    req.request_time.isoformat(),
                    req.departure_date.isoformat(),
                    req.origin.code,
                    req.destination.code,
                    req.preferred_cabin.value if req.preferred_cabin else '',
                    '',  # preferred_class doesn't exist on BookingRequest
                    req.customer.segment.value,
                    req.party_size,
                    f"{req.customer.willingness_to_pay:.2f}",
                    f"{req.customer.price_sensitivity:.2f}",
                    f"{req.customer.time_sensitivity:.2f}",
                    dtd,
                    getattr(req, 'accepted', False),
                    getattr(req, 'rejection_reason', ''),
                    loyalty_tier,
                    ancillary_prefs
                ])
        
        logger.info(f"Exported {len(self.requests)} requests to {filepath}")
        return str(filepath)
    
    def export_customers(self, filename: str = "customers.csv") -> str:
        """
        Export customer details linked to requests.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'customer_id', 'request_id', 'segment', 
                'willingness_to_pay', 'price_sensitivity', 'time_sensitivity',
                'loyalty_tier', 'ancillary_prefs'
            ])
            
            # Data
            for req in self.requests:
                if not req.customer:
                    continue
                    
                # Personalization data
                loyalty_tier = ''
                if hasattr(req.customer, 'loyalty_tier') and req.customer.loyalty_tier:
                    loyalty_tier = req.customer.loyalty_tier.value if hasattr(req.customer.loyalty_tier, 'value') else str(req.customer.loyalty_tier)
                
                ancillary_prefs = ''
                if hasattr(req.customer, 'ancillary_preferences') and req.customer.ancillary_preferences:
                    ancillary_prefs = "|".join(req.customer.ancillary_preferences)

                writer.writerow([
                    req.customer.customer_id,
                    req.request_id,
                    req.customer.segment.value,
                    f"{req.customer.willingness_to_pay:.2f}",
                    f"{req.customer.price_sensitivity:.2f}",
                    f"{req.customer.time_sensitivity:.2f}",
                    loyalty_tier,
                    ancillary_prefs
                ])
        
        logger.info(f"Exported {len(self.requests)} customers to {filepath}")
        return str(filepath)
    
    def export_raw_bookings(self, filename: str = "raw_bookings.csv") -> str:
        """
        Export detailed raw booking data including options and inventory.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'request_id', 'request_time', 'departure_date', 'days_to_departure',
                'customer_segment', 'willingness_to_pay', 'party_size',
                'accepted', 'rejection_reason',
                'chosen_price', 'chosen_class',
                'offered_options_count', 'offered_options_details',
                'inventory_at_booking'
            ])
            
            # Data
            for detail in self.detailed_requests:
                req = detail['request']
                solutions = detail['solutions']
                chosen = detail['chosen_solution']
                
                dtd = (req.departure_date - req.request_time.date()).days
                
                # Chosen details
                chosen_price = ''
                chosen_class = ''
                if chosen:
                    chosen_price = f"{chosen.total_price:.2f}"
                    if chosen.booking_classes:
                        chosen_class = chosen.booking_classes[0].value
                
                # Offered options details
                options_count = 0
                options_str = ''
                inventory_str = ''
                
                if solutions:
                    options_count = len(solutions)
                    # Create a summary string of options: "Class:Price:Seats|..."
                    opts = []
                    invs = []
                    for sol in solutions:
                        if not sol.booking_classes:
                            continue
                        bc = sol.booking_classes[0].value
                        price = sol.total_price
                        seats = sol.available_seats
                        opts.append(f"{bc}:${price:.0f}")
                        
                        # Inventory details (from the first flight in solution)
                        if sol.flights:
                            flight = sol.flights[0]
                            # Snapshot of inventory for this flight
                            # Format: Class:Booked/Cap
                            flight_inv = []
                            for cabin, cap in flight.capacity.items():
                                booked = flight.total_bookings(cabin)
                                flight_inv.append(f"{cabin.value}:{booked}/{cap}")
                            invs.append(f"{flight.schedule.flight_code}({' '.join(flight_inv)})")
                    
                    options_str = "|".join(opts)
                    inventory_str = "; ".join(set(invs)) # Unique flights
                
                writer.writerow([
                    req.request_id,
                    req.request_time.isoformat(),
                    req.departure_date.isoformat(),
                    dtd,
                    req.customer.segment.value,
                    f"{req.customer.willingness_to_pay:.2f}",
                    req.party_size,
                    detail['accepted'],
                    detail['reason'],
                    chosen_price,
                    chosen_class,
                    options_count,
                    options_str,
                    inventory_str
                ])
        
        logger.info(f"Exported {len(self.detailed_requests)} raw bookings to {filepath}")
        return str(filepath)
    
    def export_dtd_analysis(self, filename: str = "dtd_analysis.csv") -> str:
        """
        Export days-to-departure analysis.
        
        Groups bookings by DTD buckets and analyzes patterns.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        # Group by DTD
        dtd_buckets = {}
        for booking in self.bookings:
            if not booking.booking_time:
                continue
            
            # Get departure date from solution
            if not (booking.solution and booking.solution.flights):
                continue
            
            departure_date = booking.solution.flights[0].departure_date
            dtd = (departure_date - booking.booking_time.date()).days
            
            # Bucket by week
            bucket = min(dtd // 7, 12)  # 0-12+ weeks
            
            if bucket not in dtd_buckets:
                dtd_buckets[bucket] = {
                    'count': 0,
                    'revenue': 0.0,
                    'total_fare': 0.0,
                    'segments': {}
                }
            
            dtd_buckets[bucket]['count'] += 1
            dtd_buckets[bucket]['revenue'] += booking.total_revenue
            dtd_buckets[bucket]['total_fare'] += (booking.solution.total_price if booking.solution else 0)
            
            segment = booking.customer.segment.value if booking.customer else 'unknown'
            dtd_buckets[bucket]['segments'][segment] = \
                dtd_buckets[bucket]['segments'].get(segment, 0) + 1
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'weeks_before_departure', 'days_range', 'booking_count',
                'total_revenue', 'avg_fare', 'business_pct', 'leisure_pct'
            ])
            
            # Data
            for bucket in sorted(dtd_buckets.keys()):
                data = dtd_buckets[bucket]
                avg_fare = data['revenue'] / data['count'] if data['count'] > 0 else 0
                
                business_pct = (data['segments'].get('business', 0) / data['count'] * 100) \
                    if data['count'] > 0 else 0
                leisure_pct = (data['segments'].get('leisure', 0) / data['count'] * 100) \
                    if data['count'] > 0 else 0
                
                days_start = bucket * 7
                days_end = (bucket + 1) * 7 - 1
                days_range = f"{days_start}-{days_end}" if bucket < 12 else "84+"
                
                writer.writerow([
                    bucket,
                    days_range,
                    data['count'],
                    f"{data['revenue']:.2f}",
                    f"{avg_fare:.2f}",
                    f"{business_pct:.1f}",
                    f"{leisure_pct:.1f}"
                ])
        
        logger.info(f"Exported DTD analysis to {filepath}")
        return str(filepath)
    
    def export_segment_analysis(self, filename: str = "segment_analysis.csv") -> str:
        """
        Export customer segment performance analysis.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        # Group by segment
        segment_data = {}
        for booking in self.bookings:
            if not booking.customer:
                continue
                
            segment = booking.customer.segment.value
            
            if segment not in segment_data:
                segment_data[segment] = {
                    'count': 0,
                    'revenue': 0.0,
                    'total_wtp': 0.0,
                    'cabins': {},
                    'party_sizes': []
                }
            
            segment_data[segment]['count'] += 1
            segment_data[segment]['revenue'] += booking.total_revenue
            segment_data[segment]['total_wtp'] += booking.customer.willingness_to_pay
            segment_data[segment]['party_sizes'].append(booking.party_size)
            
            # Get cabin from solution
            cabin = 'unknown'
            if booking.solution and booking.solution.booking_classes:
                cabin = booking.solution.booking_classes[0].get_cabin().value
            segment_data[segment]['cabins'][cabin] = \
                segment_data[segment]['cabins'].get(cabin, 0) + 1
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'segment', 'booking_count', 'total_revenue', 'avg_revenue_per_booking',
                'avg_willingness_to_pay', 'avg_party_size', 'first_class_pct',
                'business_pct', 'economy_pct'
            ])
            
            # Data
            for segment, data in segment_data.items():
                avg_revenue = data['revenue'] / data['count'] if data['count'] > 0 else 0
                avg_wtp = data['total_wtp'] / data['count'] if data['count'] > 0 else 0
                avg_party = sum(data['party_sizes']) / len(data['party_sizes']) \
                    if data['party_sizes'] else 0
                
                first_pct = (data['cabins'].get('F', 0) / data['count'] * 100) \
                    if data['count'] > 0 else 0
                business_pct = (data['cabins'].get('J', 0) / data['count'] * 100) \
                    if data['count'] > 0 else 0
                economy_pct = (data['cabins'].get('Y', 0) / data['count'] * 100) \
                    if data['count'] > 0 else 0
                
                writer.writerow([
                    segment,
                    data['count'],
                    f"{data['revenue']:.2f}",
                    f"{avg_revenue:.2f}",
                    f"{avg_wtp:.2f}",
                    f"{avg_party:.2f}",
                    f"{first_pct:.1f}",
                    f"{business_pct:.1f}",
                    f"{economy_pct:.1f}"
                ])
        
        logger.info(f"Exported segment analysis to {filepath}")
        return str(filepath)
    
    def export_flight_metrics(self, filename: str = "flight_metrics.csv") -> str:
        """
        Export flight-level performance metrics.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        # Group bookings by flight
        flight_data = {}
        for booking in self.bookings:
            if not (booking.solution and booking.solution.flights):
                continue
                
            flight = booking.solution.flights[0]
            flight_id = flight.flight_id
            
            if flight_id not in flight_data:
                origin = flight.schedule.route.origin
                destination = flight.schedule.route.destination
                flight_data[flight_id] = {
                    'flight_code': flight.schedule.flight_code,
                    'departure_date': flight.departure_date,
                    'origin': origin.code,
                    'destination': destination.code,
                    'capacity': flight.schedule.aircraft.total_capacity,
                    'bookings': 0,
                    'revenue': 0.0,
                    'cancelled': 0
                }
            
            flight_data[flight_id]['bookings'] += 1
            flight_data[flight_id]['revenue'] += booking.total_revenue
            if booking.is_cancelled:
                flight_data[flight_id]['cancelled'] += 1
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'flight_id', 'flight_code', 'departure_date', 'origin', 'destination',
                'capacity', 'bookings', 'load_factor', 'revenue', 'revenue_per_seat',
                'avg_fare', 'cancellations'
            ])
            
            # Data
            for flight_id, data in flight_data.items():
                load_factor = (data['bookings'] / data['capacity'] * 100) \
                    if data['capacity'] > 0 else 0
                revenue_per_seat = data['revenue'] / data['capacity'] \
                    if data['capacity'] > 0 else 0
                avg_fare = data['revenue'] / data['bookings'] if data['bookings'] > 0 else 0
                
                writer.writerow([
                    flight_id,
                    data['flight_code'],
                    data['departure_date'].isoformat(),
                    data['origin'],
                    data['destination'],
                    data['capacity'],
                    data['bookings'],
                    f"{load_factor:.1f}",
                    f"{data['revenue']:.2f}",
                    f"{revenue_per_seat:.2f}",
                    f"{avg_fare:.2f}",
                    data['cancelled']
                ])
        
        logger.info(f"Exported flight metrics to {filepath}")
        return str(filepath)
    
    def export_revenue_timeline(self, filename: str = "revenue_timeline.csv") -> str:
        """
        Export revenue accumulation timeline.
        
        Returns:
            Path to created file
        """
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow(['timestamp', 'total_revenue', 'bookings_count'])
            
            # Data
            for snapshot in self.revenue_timeline:
                writer.writerow([
                    snapshot['timestamp'].isoformat(),
                    f"{snapshot['total_revenue']:.2f}",
                    snapshot['bookings_count']
                ])
        
        logger.info(f"Exported revenue timeline to {filepath}")
        return str(filepath)
    
    def export_all(self) -> Dict[str, str]:
        """
        Export all data to CSV files.
        
        Returns:
            Dictionary mapping data type to file path
        """
        exports = {}
        
        if self.bookings:
            exports['bookings'] = self.export_bookings()
            exports['dtd_analysis'] = self.export_dtd_analysis()
            exports['segment_analysis'] = self.export_segment_analysis()
            exports['flight_metrics'] = self.export_flight_metrics()
        
        if self.requests:
            exports['requests'] = self.export_requests()
            exports['customers'] = self.export_customers()
            exports['raw_bookings'] = self.export_raw_bookings()
        
        if self.revenue_timeline:
            exports['revenue_timeline'] = self.export_revenue_timeline()
        
        logger.info(f"Exported {len(exports)} data files to {self.output_dir}")
        return exports
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of collected data."""
        return {
            'bookings_count': len(self.bookings),
            'requests_count': len(self.requests),
            'flight_snapshots': len(self.flight_snapshots),
            'revenue_snapshots': len(self.revenue_timeline),
            'output_directory': str(self.output_dir)
        }
