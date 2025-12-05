
import sys
import os
import logging
from datetime import date
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.getcwd())

from core.simulator import Simulator, SimulationConfig
from demand.forecaster import DemandForecaster, ForecastMethod
from demand.generator import MultiStreamDemandGenerator, generate_default_holidays
from competition.market import Market
from examples.competitive_simulation import (
    create_competitive_network,
    create_airlines,
    create_competitive_demand
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_simulation_direct")

def run_direct_simulation():
    print("Starting direct simulation (1 year)...")
    
    # Configuration
    start_date = date(2025, 1, 1)
    end_date = date(2025, 12, 31)
    sim_id = "direct_sim_year"
    output_dir = f"simulation_results/{sim_id}"
    os.makedirs(output_dir, exist_ok=True)
    
    config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        random_seed=42,
        rm_method="EMSR-b",
        rm_optimization_frequency=24,
        forecast_method="pickup",
        choice_model="mnl",
        dynamic_pricing=True,
        overbooking_enabled=True,
        optimization_horizons=[30, 14, 7, 3, 1],
        price_update_frequency_hours=6.0,
        demand_generation_method="poisson",
        overbooking_method="critical_fractile",
        overbooking_risk_tolerance=0.05,
        include_buyup_down=True,
        include_recapture=True,
        personalization_enabled=False,
        use_db=True,
        db_path=f"{output_dir}/bookings.db",
        progress_bar=True,
        export_csv=True,
        output_dir=output_dir
    )

    # Network & Airlines
    network = create_competitive_network()
    airlines = create_airlines(network['routes'])
    
    # Filter for AA and JFK-LAX
    selected_airline = "AA"
    airlines = {k: v for k, v in airlines.items() if k == selected_airline}
    
    network['routes'] = {k: v for k, v in network['routes'].items() if k == 'JFK-LAX'}
    for code, airline in airlines.items():
        airline.schedules = [s for s in airline.schedules if s.route.origin.code == 'JFK' and s.route.destination.code == 'LAX']

    # Market
    market = Market(information_transparency=0.75)
    for airline in airlines.values():
        market.add_airline(airline)

    # Forecasters
    forecasters = {
        code: DemandForecaster(method=ForecastMethod.PICKUP, track_accuracy=True, add_noise=True, noise_std=0.1)
        for code in airlines.keys()
    }

    # Demand
    full_network = create_competitive_network()
    demand_streams = create_competitive_demand(full_network['routes'])
    demand_streams = [s for s in demand_streams if s.origin.code == 'JFK' and s.destination.code == 'LAX']
    
    # Holidays
    years = list(range(start_date.year, end_date.year + 1))
    holidays = generate_default_holidays(years)
    print(f"Generated {len(holidays)} holiday dates.")

    # Apply demand parameters
    for stream in demand_streams:
        stream.holidays = holidays
        stream.mean_daily_demand = 100.0 # Default
        stream.demand_std = 20.0
        stream.business_proportion = 0.30
        stream.business_wtp_mean = 800.0
        stream.leisure_wtp_mean = 300.0

    demand_generator = MultiStreamDemandGenerator(demand_streams)
    print("Generating demand...")
    all_requests = demand_generator.generate_all_requests(start_date, end_date)
    print(f"Generated {len(all_requests)} booking requests.")

    # Run Simulation
    for airline_code, airline in airlines.items():
        print(f"Simulating {airline.name}...")
        
        simulator = Simulator(
            config=config,
            schedules=airline.schedules,
            routes=list(network['routes'].values()),
            airports=network['airports']
        )
        
        # Inject dependencies
        simulator.market = market
        simulator.forecaster = forecasters[airline_code]
        
        # Inject demand
        demand_generator.add_requests_to_event_queue(
            simulator.event_manager,
            all_requests
        )
        
        # Run
        results = simulator.run()
        
        print(f"Simulation complete. Revenue: ${results.total_revenue:,.2f}")
        
        # The simulator exports CSVs if export_csv=True in config
        # We need to move/copy the bookings.csv to where analyze_dtd_buckets.py expects it
        # Or update analyze_dtd_buckets.py to point to the new location.
        
        src_csv = f"{output_dir}/bookings.csv"
        dest_csv = "simulation_results/downloaded_bookings.csv"
        
        if os.path.exists(src_csv):
            import shutil
            shutil.copy(src_csv, dest_csv)
            print(f"Copied results to {dest_csv}")
        else:
            print(f"Warning: {src_csv} not found.")

if __name__ == "__main__":
    run_direct_simulation()
