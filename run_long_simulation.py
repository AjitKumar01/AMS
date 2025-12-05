"""
Long duration simulation to verify seasonality across multiple holidays.
Runs for 6 months (Jan - Jun 2025).
"""

from datetime import date, time, timedelta
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from core.models import (
    Airport, Route, Aircraft, FlightSchedule, CabinClass, BookingClass
)
from core.simulator import Simulator, SimulationConfig
from demand.generator import DemandStreamConfig, generate_default_holidays, MultiStreamDemandGenerator

def create_network():
    jfk = Airport("JFK", "New York", "New York", "USA", "EST", 40.6, -73.7)
    lax = Airport("LAX", "Los Angeles", "Los Angeles", "USA", "PST", 33.9, -118.4)
    route = Route(jfk, lax, 3983)
    return [jfk, lax], [route]

def create_schedule(airports, routes):
    jfk, lax = airports
    route = routes[0]
    
    aircraft = Aircraft(
        type_code="B777",
        name="Boeing 777",
        capacity={
            CabinClass.FIRST: 8,
            CabinClass.BUSINESS: 52,
            CabinClass.ECONOMY: 220
        }
    )
    
    # Daily flight
    schedule = FlightSchedule(
        airline_code="AA",
        flight_number="100",
        route=route,
        departure_time=time(8, 0),
        arrival_time=time(11, 30),
        days_of_week=[0, 1, 2, 3, 4, 5, 6],
        aircraft=aircraft,
        valid_from=date(2025, 1, 1),
        valid_until=date(2025, 12, 31)
    )
    return [schedule]

def run_simulation():
    print("="*50)
    print("Running 6-Month Seasonality Simulation")
    print("="*50)
    
    airports, routes = create_network()
    schedules = create_schedule(airports, routes)
    
    # Generate holidays
    holidays = generate_default_holidays([2025])
    
    # Demand Config
    demand_config = DemandStreamConfig(
        stream_id="JFK-LAX",
        origin=airports[0],
        destination=airports[1],
        mean_daily_demand=150.0,
        demand_std=20.0,
        holidays=holidays
    )
    
    # Simulation Config (Jan 1 to Jun 30)
    sim_config = SimulationConfig(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 6, 30),
        random_seed=42,
        rm_method="EMSR-b",
        db_path="simulation_results/seasonality_sim.db"
    )
    
    # Initialize Simulator
    simulator = Simulator(
        config=sim_config,
        schedules=schedules,
        routes=routes,
        airports=airports
    )
    
    # Generate Demand
    print("Generating demand...")
    demand_gen = MultiStreamDemandGenerator([demand_config], random_seed=42)
    requests = demand_gen.generate_all_requests(
        start_date=sim_config.start_date,
        end_date=sim_config.end_date
    )
    print(f"Generated {len(requests)} booking requests")
    
    # Add requests to simulator
    demand_gen.add_requests_to_event_queue(simulator.event_manager, requests)
    
    print("Starting simulation...")
    simulator.run()
    print("Simulation complete.")

if __name__ == "__main__":
    os.makedirs("simulation_results", exist_ok=True)
    run_simulation()
