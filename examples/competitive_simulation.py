"""
Advanced competitive simulation example.

Demonstrates:
- Multi-airline competition (3 airlines with different strategies)
- Network revenue management (O-D control with displacement costs)
- ML-based demand forecasting with accuracy tracking
- Realistic forecast errors and their impact on RM performance
"""

import sys
from pathlib import Path
from datetime import date, time, timedelta
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    Airport, Route, Aircraft, FlightSchedule, CabinClass, BookingClass
)
from core.simulator import Simulator, SimulationConfig
from core.events import EventManager

from demand.generator import DemandStreamConfig, MultiStreamDemandGenerator
from demand.forecaster import DemandForecaster, ForecastMethod

from competition.airline import Airline, CompetitiveStrategy
from competition.market import Market

from inventory.network import NetworkOptimizer


def create_competitive_network():
    """
    Create a competitive network with 3 airlines.
    
    Market structure:
    - JFK-LAX: Hub route, all 3 airlines compete
    - JFK-ORD: American and United compete
    - ORD-LAX: American and United compete
    - LAX-SFO: Delta and United compete
    """
    # Airports
    jfk = Airport(code="JFK", name="John F. Kennedy International", 
                  city="New York", country="USA", timezone="America/New_York",
                  lat=40.6413, lon=-73.7781)
    
    lax = Airport(code="LAX", name="Los Angeles International",
                  city="Los Angeles", country="USA", timezone="America/Los_Angeles",
                  lat=33.9416, lon=-118.4085)
    
    ord_airport = Airport(code="ORD", name="O'Hare International",
                         city="Chicago", country="USA", timezone="America/Chicago",
                         lat=41.9742, lon=-87.9073)
    
    sfo = Airport(code="SFO", name="San Francisco International",
                  city="San Francisco", country="USA", timezone="America/Los_Angeles",
                  lat=37.6213, lon=-122.3790)
    
    # Routes (distances in km)
    routes = {
        'JFK-LAX': Route(origin=jfk, destination=lax, distance_km=3983),  # ~2475 miles
        'JFK-ORD': Route(origin=jfk, destination=ord_airport, distance_km=1191),  # ~740 miles
        'ORD-LAX': Route(origin=ord_airport, destination=lax, distance_km=2808),  # ~1745 miles
        'LAX-SFO': Route(origin=lax, destination=sfo, distance_km=542)  # ~337 miles
    }
    
    return {
        'airports': [jfk, lax, ord_airport, sfo],
        'routes': routes
    }


def create_airlines(routes: Dict[str, Route]) -> Dict[str, Airline]:
    """
    Create 3 competing airlines with different strategies.
    """
    # Aircraft types
    boeing_777 = Aircraft(
        type_code="B772",
        name="Boeing 777-200",
        capacity={
            CabinClass.FIRST: 8,
            CabinClass.BUSINESS: 52,
            CabinClass.PREMIUM_ECONOMY: 40,
            CabinClass.ECONOMY: 200
        }
    )
    
    airbus_a330 = Aircraft(
        type_code="A333",
        name="Airbus A330-300",
        capacity={
            CabinClass.FIRST: 0,
            CabinClass.BUSINESS: 45,
            CabinClass.PREMIUM_ECONOMY: 35,
            CabinClass.ECONOMY: 220
        }
    )
    
    # American Airlines - Aggressive strategy
    american = Airline(
        code="AA",
        name="American Airlines",
        strategy=CompetitiveStrategy.AGGRESSIVE,
        base_price_multiplier=0.95,  # Price below market
        quality_multiplier=1.0,
        cost_per_seat_mile=0.10,
        brand_preference=0.1
    )
    
    # Schedules for American
    aa_schedules = [
        FlightSchedule(
            airline_code="AA",
            flight_number="100",
            route=routes['JFK-LAX'],
            departure_time=time(8, 0),
            arrival_time=time(11, 30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=boeing_777,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        ),
        FlightSchedule(
            airline_code="AA",
            flight_number="200",
            route=routes['JFK-ORD'],
            departure_time=time(7, 0),
            arrival_time=time(8, 30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=airbus_a330,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        ),
        FlightSchedule(
            airline_code="AA",
            flight_number="300",
            route=routes['ORD-LAX'],
            departure_time=time(10, 0),
            arrival_time=time(12, 30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=boeing_777,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        )
    ]
    
    for schedule in aa_schedules:
        american.add_schedule(schedule)
    
    # United Airlines - ML-based strategy
    united = Airline(
        code="UA",
        name="United Airlines",
        strategy=CompetitiveStrategy.ML_BASED,
        base_price_multiplier=1.0,  # Market price
        quality_multiplier=1.05,
        cost_per_seat_mile=0.11,
        brand_preference=0.05
    )
    
    ua_schedules = [
        FlightSchedule(
            airline_code="UA",
            flight_number="800",
            route=routes['JFK-LAX'],
            departure_time=time(9, 30),
            arrival_time=time(13, 0),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=boeing_777,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        ),
        FlightSchedule(
            airline_code="UA",
            flight_number="900",
            route=routes['JFK-ORD'],
            departure_time=time(13, 0),
            arrival_time=time(14, 30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=airbus_a330,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        ),
        FlightSchedule(
            airline_code="UA",
            flight_number="1000",
            route=routes['LAX-SFO'],
            departure_time=time(16, 0),
            arrival_time=time(17, 30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=airbus_a330,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        )
    ]
    
    for schedule in ua_schedules:
        united.add_schedule(schedule)
    
    # Delta - Conservative/Premium strategy
    delta = Airline(
        code="DL",
        name="Delta Air Lines",
        strategy=CompetitiveStrategy.CONSERVATIVE,
        base_price_multiplier=1.10,  # Premium pricing
        quality_multiplier=1.15,
        cost_per_seat_mile=0.12,
        brand_preference=0.15
    )
    
    dl_schedules = [
        FlightSchedule(
            airline_code="DL",
            flight_number="500",
            route=routes['JFK-LAX'],
            departure_time=time(14, 0),
            arrival_time=time(17, 30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=boeing_777,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        ),
        FlightSchedule(
            airline_code="DL",
            flight_number="600",
            route=routes['LAX-SFO'],
            departure_time=time(11, 0),
            arrival_time=time(12, 30),
            days_of_week=[0, 1, 2, 3, 4, 5, 6],
            aircraft=airbus_a330,
            valid_from=date(2025, 12, 1),
            valid_until=date(2025, 12, 31)
        )
    ]
    
    for schedule in dl_schedules:
        delta.add_schedule(schedule)
    
    return {
        'AA': american,
        'UA': united,
        'DL': delta
    }


def create_competitive_demand(routes: Dict[str, Route]):
    """
    Create demand streams for competitive markets.
    """
    demand_streams = []
    
    # JFK-LAX: High demand, competitive market
    demand_streams.append(DemandStreamConfig(
        stream_id="JFK-LAX-Main",
        origin=routes['JFK-LAX'].origin,
        destination=routes['JFK-LAX'].destination,
        mean_daily_demand=300.0,  # High demand
        business_proportion=0.40,  # Many business travelers
        business_wtp_mean=1200.0,
        business_wtp_std=300.0,
        leisure_wtp_mean=450.0,
        leisure_wtp_std=150.0,
        mean_advance_purchase=25.0,
        max_advance_days=90
    ))
    
    # JFK-ORD: Moderate demand
    demand_streams.append(DemandStreamConfig(
        stream_id="JFK-ORD-Main",
        origin=routes['JFK-ORD'].origin,
        destination=routes['JFK-ORD'].destination,
        mean_daily_demand=200.0,
        business_proportion=0.50,  # High business
        business_wtp_mean=800.0,
        business_wtp_std=200.0,
        leisure_wtp_mean=300.0,
        leisure_wtp_std=100.0,
        mean_advance_purchase=20.0,
        max_advance_days=60
    ))
    
    # ORD-LAX: Connecting traffic
    demand_streams.append(DemandStreamConfig(
        stream_id="ORD-LAX-Main",
        origin=routes['ORD-LAX'].origin,
        destination=routes['ORD-LAX'].destination,
        mean_daily_demand=180.0,
        business_proportion=0.35,
        business_wtp_mean=900.0,
        business_wtp_std=250.0,
        leisure_wtp_mean=350.0,
        leisure_wtp_std=120.0,
        mean_advance_purchase=22.0,
        max_advance_days=75
    ))
    
    # LAX-SFO: Short haul, high frequency
    demand_streams.append(DemandStreamConfig(
        stream_id="LAX-SFO-Main",
        origin=routes['LAX-SFO'].origin,
        destination=routes['LAX-SFO'].destination,
        mean_daily_demand=150.0,
        business_proportion=0.60,  # Very high business
        business_wtp_mean=400.0,
        business_wtp_std=100.0,
        leisure_wtp_mean=180.0,
        leisure_wtp_std=50.0,
        mean_advance_purchase=10.0,
        max_advance_days=30
    ))
    
    return demand_streams


def run_competitive_simulation():
    """
    Run competitive simulation with all advanced features.
    """
    print("=" * 70)
    print("PyAirline RM - Advanced Competitive Simulation")
    print("=" * 70)
    print()
    
    # Step 1: Create network
    print("Step 1: Creating competitive network...")
    network = create_competitive_network()
    print(f"  - Airports: {len(network['airports'])}")
    print(f"  - Routes: {len(network['routes'])}")
    print()
    
    # Step 2: Create airlines
    print("Step 2: Creating airlines with different strategies...")
    airlines = create_airlines(network['routes'])
    for code, airline in airlines.items():
        print(f"  - {airline.name} ({code}): {airline.strategy.value}")
        print(f"    Flights: {len(airline.schedules)}, "
              f"Price mult: {airline.base_price_multiplier:.2f}")
    print()
    
    # Step 3: Create market coordinator
    print("Step 3: Initializing market coordinator...")
    market = Market(information_transparency=0.75)  # 75% transparency
    for airline in airlines.values():
        market.add_airline(airline)
    print(f"  - Information transparency: 75%")
    print(f"  - Airlines: {len(market.airlines)}")
    print()
    
    # Step 4: Setup demand forecasting with different accuracy levels
    print("Step 4: Configuring demand forecasting...")
    forecasters = {
        'AA': DemandForecaster(
            method=ForecastMethod.PICKUP,
            track_accuracy=True,
            add_noise=True,
            noise_std=0.15  # 15% forecast error
        ),
        'UA': DemandForecaster(
            method=ForecastMethod.NEURAL_NETWORK,
            track_accuracy=True,
            add_noise=True,
            noise_std=0.08  # 8% forecast error (better ML)
        ),
        'DL': DemandForecaster(
            method=ForecastMethod.EXPONENTIAL_SMOOTHING,
            track_accuracy=True,
            add_noise=True,
            noise_std=0.12  # 12% forecast error
        )
    }
    
    for code, forecaster in forecasters.items():
        print(f"  - {code}: {forecaster.method.value} "
              f"(error std: {forecaster.noise_std*100:.0f}%)")
    print()
    
    # Step 5: Setup network RM optimizers
    print("Step 5: Configuring network revenue management...")
    network_optimizers = {}
    for code in airlines.keys():
        optimizer = NetworkOptimizer(
            num_virtual_buckets=8,
            optimization_method="linear_programming"
        )
        network_optimizers[code] = optimizer
        print(f"  - {code}: O-D control with {optimizer.num_virtual_buckets} buckets")
    print()
    
    # Step 6: Create demand
    print("Step 6: Generating competitive demand...")
    demand_streams = create_competitive_demand(network['routes'])
    demand_generator = MultiStreamDemandGenerator(demand_streams)
    
    start_date = date(2025, 12, 1)
    end_date = date(2025, 12, 31)
    
    all_requests = demand_generator.generate_all_requests(start_date, end_date)
    print(f"  - Total booking requests: {len(all_requests):,}")
    print(f"  - Average per day: {len(all_requests) / 31:.1f}")
    print()
    
    # Step 7: Configure and run simulation
    print("Step 7: Configuring simulation...")
    config = SimulationConfig(
        start_date=start_date,
        end_date=end_date,
        rm_method="EMSR-b",
        optimization_horizons=[60, 45, 30, 21, 14, 7, 3, 1],
        dynamic_pricing=True,
        overbooking_enabled=True,
        choice_model="mnl"
    )
    print(f"  - Period: {start_date} to {end_date}")
    print(f"  - Dynamic pricing: {config.dynamic_pricing}")
    print(f"  - Optimization horizons: {len(config.optimization_horizons)}")
    print()
    
    print("=" * 70)
    print("Running Simulation (this will take a moment)...")
    print("=" * 70)
    print()
    
    # Create simulator for each airline
    # Note: In a full implementation, we'd have a CompetitiveSimulator
    # that coordinates all airlines. For now, we'll run them sequentially
    # and use the market coordinator to share information.
    
    results_by_airline = {}
    
    for airline_code, airline in airlines.items():
        print(f"Simulating {airline.name}...")
        
        # Create simulator for this airline
        simulator = Simulator(
            config=config,
            schedules=airline.schedules,
            routes=list(network['routes'].values()),
            airports=network['airports']
        )
        
        # Add demand (in reality, demand would be split among airlines based on choice)
        # For simplicity, we'll give each airline equal share of demand
        # In reality, demand would be generated per airline based on market share
        demand_generator.add_requests_to_event_queue(
            simulator.event_manager,
            all_requests
        )
        
        # Run simulation
        results = simulator.run()
        results_by_airline[airline_code] = results
        
        # Evaluate forecast accuracy (post-simulation)
        # We generate a forecast for DTD=30 and compare to final bookings
        if airline_code in forecasters:
            forecaster = forecasters[airline_code]
            for flight in simulator.flight_dates.values():
                forecast_date = flight.departure_date - timedelta(days=30)
                forecast = forecaster.forecast_demand(
                    flight_date=flight,
                    current_date=forecast_date,
                    current_bookings={} 
                )
                forecaster.evaluate_forecast(forecast, flight.bookings)
        
        # Record in market
        for booking in results.bookings:
            market.record_booking(
                airline_code=airline_code,
                origin=booking.solution.flights[0].schedule.route.origin,
                destination=booking.solution.flights[-1].schedule.route.destination,
                passengers=booking.party_size,
                revenue=booking.total_revenue
            )
        
        # Share competitive intelligence
        market.share_competitive_intelligence(end_date)
        
        print(f"  ✓ Completed - Revenue: ${results.total_revenue:,.0f}, "
              f"Bookings: {results.total_bookings:,}")
        print()
    
    # Step 8: Analyze results
    print("=" * 70)
    print("Competitive Market Analysis")
    print("=" * 70)
    print()
    
    # Overall results
    print("Airline Performance Comparison:")
    print("-" * 70)
    print(f"{'Airline':<15} {'Revenue':>12} {'Bookings':>10} {'Load Factor':>12} {'Avg Fare':>10}")
    print("-" * 70)
    
    for airline_code, results in results_by_airline.items():
        airline = airlines[airline_code]
        print(f"{airline.name:<15} ${results.total_revenue:>11,.0f} "
              f"{results.total_bookings:>10,} {results.load_factor:>11.1%} "
              f"${results.average_fare:>9,.0f}")
    
    print()
    
    # Market share analysis
    print("Market Share Analysis by Route:")
    print("-" * 70)
    
    competitive_analysis = market.analyze_competitive_dynamics()
    
    for segment_key, segment in market.segments.items():
        summary = market.get_market_summary(segment_key)
        print(f"\n{segment_key}:")
        print(f"  Total passengers: {summary['total_demand']:.0f}")
        print(f"  Total revenue: ${summary['total_revenue']:,.0f}")
        print(f"  HHI: {summary['hhi']:.0f} ({summary.get('concentration', 'N/A')})")
        print(f"  Competitors: {summary['num_competitors']}")
        
        if 'airlines' in summary:
            print(f"  Market shares:")
            for al_code, al_data in summary['airlines'].items():
                print(f"    {al_code}: {al_data['market_share_pax']:.1%} pax, "
                      f"{al_data['market_share_revenue']:.1%} revenue, "
                      f"avg fare ${al_data['average_fare']:.0f}")
    
    print()
    
    # Forecast accuracy analysis
    print("Demand Forecasting Accuracy:")
    print("-" * 70)
    
    for airline_code, forecaster in forecasters.items():
        report = forecaster.get_accuracy_report()
        print(f"\n{airlines[airline_code].name} ({report['method']}):")
        print(f"  MAE: {report['mae']:.2f} passengers")
        print(f"  MAPE: {report['mape']:.1f}%")
        print(f"  Bias: {report['bias']:.2f} (+ = over-forecast)")
        print(f"  Revenue lost to errors: ${report['revenue_lost']:,.0f}")
        print(f"  Impact on total revenue: "
              f"{report['revenue_lost'] / results_by_airline[airline_code].total_revenue * 100:.2f}%")
    
    print()
    
    # Network RM analysis
    print("Network Revenue Management Performance:")
    print("-" * 70)
    
    for airline_code, optimizer in network_optimizers.items():
        report = optimizer.get_displacement_report()
        if report['total_legs'] > 0:
            print(f"\n{airlines[airline_code].name}:")
            print(f"  Legs managed: {report['total_legs']}")
            print(f"  Avg bid price: ${report['avg_bid_price']:.2f}")
            print(f"  Max bid price: ${report['max_bid_price']:.2f}")
            print(f"  Min bid price: ${report['min_bid_price']:.2f}")
    
    print()
    print("=" * 70)
    print("Simulation Complete!")
    print("=" * 70)
    print()
    
    # Key insights
    print("Key Insights:")
    print("-" * 70)
    
    # Winner by revenue
    winner = max(results_by_airline.items(), key=lambda x: x[1].total_revenue)
    print(f"1. Revenue leader: {airlines[winner[0]].name} "
          f"(${winner[1].total_revenue:,.0f})")
    
    # Best load factor
    best_lf = max(results_by_airline.items(), key=lambda x: x[1].load_factor)
    print(f"2. Best load factor: {airlines[best_lf[0]].name} "
          f"({best_lf[1].load_factor:.1%})")
    
    # Best forecasting
    best_forecast = min(forecasters.items(), 
                       key=lambda x: x[1].get_accuracy_report()['mae'])
    print(f"3. Most accurate forecasts: {airlines[best_forecast[0]].name} "
          f"(MAE: {best_forecast[1].get_accuracy_report()['mae']:.2f})")
    
    # Market concentration
    print(f"4. Overall market HHI: {competitive_analysis['overall_hhi']:.0f}")
    if competitive_analysis['overall_hhi'] < 1500:
        print(f"   → Competitive market (healthy competition)")
    elif competitive_analysis['overall_hhi'] < 2500:
        print(f"   → Moderately concentrated market")
    else:
        print(f"   → Highly concentrated market (limited competition)")
    
    print()

    # Step 9: Demonstrate Hybrid Forecasting (Unconstraining)
    print("=" * 70)
    print("Step 9: Hybrid Forecasting (FRAT5 Unconstraining) Demo")
    print("=" * 70)
    print("Demonstrating how to unconstrain demand for a closed booking class...")
    
    # Create a forecaster instance
    demo_forecaster = DemandForecaster()
    
    # Simulate a scenario where Economy (Y) was closed due to high demand,
    # but Business (J) remained open.
    print("\nScenario: Flight DL500 (JFK-LAX)")
    print(" - Economy Class (Y): CLOSED (Fare $200)")
    print(" - Business Class (J): OPEN (Fare $500)")
    print(" - Observed Bookings: Y=0 (Closed), J=20")
    
    observed = {
        BookingClass.Y: 0,
        BookingClass.J: 20
    }
    
    fares = {
        BookingClass.Y: 200.0,
        BookingClass.J: 500.0
    }
    
    availability = {
        BookingClass.Y: False,
        BookingClass.J: True
    }
    
    # Run unconstraining
    unconstrained = demo_forecaster.unconstrain_demand_with_frat5(
        observed_bookings=observed,
        fares=fares,
        availability=availability
    )
    
    print("\nResults:")
    print(f" - Observed Demand: {observed}")
    print(f" - Unconstrained Demand: {unconstrained}")
    
    added_demand = unconstrained[BookingClass.Y] - observed[BookingClass.Y]
    print(f" - Estimated Lost Demand for Class Y: {added_demand:.1f} pax")
    print("   (These are customers who wanted Y but refused to pay for J)")
    print()
    
    return {
        'airlines': airlines,
        'results': results_by_airline,
        'market': market,
        'forecasters': forecasters,
        'network_optimizers': network_optimizers
    }


if __name__ == "__main__":
    simulation_data = run_competitive_simulation()
