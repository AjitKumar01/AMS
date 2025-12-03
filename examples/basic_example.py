"""
Example: Basic single-airline revenue management simulation.

This demonstrates:
- Setting up flight schedules
- Generating demand
- Running RM optimization
- Analyzing results
"""

from datetime import date, time, timedelta
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.models import (
    Airport, Route, Aircraft, FlightSchedule, CabinClass, BookingClass
)
from core.simulator import Simulator, SimulationConfig
from demand.generator import DemandStreamConfig, MultiStreamDemandGenerator
from rm.optimizer import RMOptimizer, DemandForecast

def create_sample_network():
    """Create a simple 3-airport network."""
    
    # Define airports
    jfk = Airport(
        code="JFK",
        name="John F. Kennedy International Airport",
        city="New York",
        country="USA",
        timezone="America/New_York",
        lat=40.6413,
        lon=-73.7781
    )
    
    lax = Airport(
        code="LAX",
        name="Los Angeles International Airport",
        city="Los Angeles",
        country="USA",
        timezone="America/Los_Angeles",
        lat=33.9416,
        lon=-118.4085
    )
    
    ord = Airport(
        code="ORD",
        name="O'Hare International Airport",
        city="Chicago",
        country="USA",
        timezone="America/Chicago",
        lat=41.9742,
        lon=-87.9073
    )
    
    # Define routes
    jfk_lax = Route(jfk, lax, 3983)  # km
    ord_lax = Route(ord, lax, 2802)
    jfk_ord = Route(jfk, ord, 1185)
    
    return [jfk, lax, ord], [jfk_lax, ord_lax, jfk_ord]


def create_sample_schedules(airports, routes):
    """Create flight schedules."""
    
    jfk, lax, ord = airports
    jfk_lax, ord_lax, jfk_ord = routes
    
    # Define aircraft
    b777 = Aircraft(
        type_code="B777",
        name="Boeing 777-300ER",
        capacity={
            CabinClass.FIRST: 8,
            CabinClass.BUSINESS: 52,
            CabinClass.ECONOMY: 220
        }
    )
    
    a320 = Aircraft(
        type_code="A320",
        name="Airbus A320",
        capacity={
            CabinClass.BUSINESS: 20,
            CabinClass.ECONOMY: 130
        }
    )
    
    # Create schedules
    schedules = []
    
    # AA100: JFK-LAX daily morning flight
    schedules.append(FlightSchedule(
        airline_code="AA",
        flight_number="100",
        route=jfk_lax,
        departure_time=time(8, 0),
        arrival_time=time(11, 30),
        days_of_week=[0, 1, 2, 3, 4, 5, 6],  # Daily
        aircraft=b777,
        valid_from=date(2025, 1, 1),
        valid_until=date(2025, 12, 31)
    ))
    
    # AA101: JFK-LAX daily evening flight
    schedules.append(FlightSchedule(
        airline_code="AA",
        flight_number="101",
        route=jfk_lax,
        departure_time=time(17, 0),
        arrival_time=time(20, 30),
        days_of_week=[0, 1, 2, 3, 4, 5, 6],
        aircraft=b777,
        valid_from=date(2025, 1, 1),
        valid_until=date(2025, 12, 31)
    ))
    
    # AA200: ORD-LAX daily
    schedules.append(FlightSchedule(
        airline_code="AA",
        flight_number="200",
        route=ord_lax,
        departure_time=time(10, 0),
        arrival_time=time(12, 30),
        days_of_week=[0, 1, 2, 3, 4, 5, 6],
        aircraft=a320,
        valid_from=date(2025, 1, 1),
        valid_until=date(2025, 12, 31)
    ))
    
    return schedules


def create_demand_config(airports):
    """Create demand stream configurations."""
    
    jfk, lax, ord = airports
    
    # JFK-LAX demand (high volume transcontinental)
    jfk_lax_demand = DemandStreamConfig(
        stream_id="JFK-LAX",
        origin=jfk,
        destination=lax,
        mean_daily_demand=150.0,  # High demand route
        demand_std=30.0,
        business_proportion=0.35,  # 35% business
        business_wtp_mean=900.0,
        business_wtp_std=250.0,
        leisure_wtp_mean=350.0,
        leisure_wtp_std=120.0,
        mean_advance_purchase=25.0
    )
    
    # ORD-LAX demand (moderate volume)
    ord_lax_demand = DemandStreamConfig(
        stream_id="ORD-LAX",
        origin=ord,
        destination=lax,
        mean_daily_demand=80.0,
        demand_std=20.0,
        business_proportion=0.30,
        business_wtp_mean=700.0,
        business_wtp_std=200.0,
        leisure_wtp_mean=300.0,
        leisure_wtp_std=100.0,
        mean_advance_purchase=21.0
    )
    
    return [jfk_lax_demand, ord_lax_demand]


def main():
    """Run the simulation."""
    
    print("="*70)
    print("PyAirline RM - Basic Example Simulation")
    print("="*70)
    print()
    
    # Step 1: Create network
    print("Step 1: Creating network...")
    airports, routes = create_sample_network()
    schedules = create_sample_schedules(airports, routes)
    print(f"  - Airports: {len(airports)}")
    print(f"  - Routes: {len(routes)}")
    print(f"  - Schedules: {len(schedules)}")
    print()
    
    # Step 2: Configure simulation
    print("Step 2: Configuring simulation...")
    config = SimulationConfig(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),  # 1 month
        random_seed=42,
        rm_optimization_frequency="daily",
        rm_method="EMSR-b",
        optimization_horizons=[30, 14, 7, 3],
        dynamic_pricing=False,  # Start simple
        snapshot_frequency_days=7,
        progress_bar=True,
        log_level="INFO"
    )
    print(f"  - Period: {config.start_date} to {config.end_date}")
    print(f"  - RM Method: {config.rm_method}")
    print()
    
    # Step 3: Generate demand
    print("Step 3: Generating demand...")
    demand_configs = create_demand_config(airports)
    demand_gen = MultiStreamDemandGenerator(demand_configs, random_seed=42)
    
    requests = demand_gen.generate_all_requests(
        start_date=config.start_date,
        end_date=config.end_date
    )
    print(f"  - Generated {len(requests)} booking requests")
    print(f"  - Average per day: {len(requests) / 31:.1f}")
    print()
    
    # Step 4: Create simulator
    print("Step 4: Initializing simulator...")
    simulator = Simulator(
        config=config,
        schedules=schedules,
        routes=routes,
        airports=airports
    )
    print(f"  - Flight dates created: {len(simulator.flight_dates)}")
    print()
    
    # Step 5: Add demand to event queue
    print("Step 5: Adding demand to event queue...")
    num_events = demand_gen.add_requests_to_event_queue(
        simulator.event_manager,
        requests
    )
    print(f"  - Added {num_events} booking request events")
    print(f"  - Total events in queue: {simulator.event_manager.size()}")
    print()
    
    # Step 6: Run simulation
    print("Step 6: Running simulation...")
    print("-"*70)
    results = simulator.run()
    print("-"*70)
    print()
    
    # Step 7: Analyze results
    print("Step 7: Results Summary")
    print("="*70)
    print(results.summary())
    
    # Additional analysis
    print("\nDetailed Metrics:")
    print(f"  Booking Success Rate: {(results.total_bookings / len(requests)) * 100:.1f}%")
    print(f"  Revenue per Flight: ${results.total_revenue / results.flights_operated:,.2f}")
    print(f"  Revenue per Booking: ${results.total_revenue / max(1, results.total_bookings):,.2f}")
    print(f"  Average Party Size: {results.total_seats_sold / max(1, results.total_bookings):.2f}")
    
    # Demand statistics
    demand_stats = demand_gen.get_total_statistics()
    print(f"\nDemand Generation:")
    print(f"  Total Requests Generated: {demand_stats['total_requests']}")
    print(f"  Number of Streams: {demand_stats['num_streams']}")
    for stream_id, stats in demand_stats['streams'].items():
        print(f"  {stream_id}:")
        print(f"    - Total: {stats['total_requests']}")
        print(f"    - By Segment: {stats['by_segment']}")
    
    print()
    print("="*70)
    print("Simulation Complete!")
    print("="*70)
    
    return results


if __name__ == "__main__":
    results = main()
