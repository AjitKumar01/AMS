# PyAirline RM - Getting Started Guide

## Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- (Optional) virtualenv or conda for isolated environment

### Step 1: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv pyairline_env
source pyairline_env/bin/activate  # On Windows: pyairline_env\Scripts\activate

# Or using conda
conda create -n pyairline python=3.10
conda activate pyairline
```

### Step 2: Install Dependencies

```bash
cd pyairline_rm
pip install -r requirements.txt
```

### Step 3: Install Package

```bash
# Development installation (editable)
pip install -e .

# Or production installation
pip install .
```

## Quick Start

### Running the Basic Example

The simplest way to understand the simulator is to run the basic example:

```bash
python examples/basic_example.py
```

This will:
1. Create a simple 3-airport network (JFK, LAX, ORD)
2. Define flight schedules for American Airlines
3. Generate realistic passenger demand
4. Run a 1-month simulation with RM optimization
5. Display results and statistics

### Understanding the Output

The example will output:

```
==================================================================
PyAirline RM - Basic Example Simulation
==================================================================

Step 1: Creating network...
  - Airports: 3
  - Routes: 3
  - Schedules: 3

Step 2: Configuring simulation...
  - Period: 2025-01-01 to 2025-01-31
  - RM Method: EMSR-b

Step 3: Generating demand...
  - Generated 7,130 booking requests
  - Average per day: 230.0

[... simulation runs ...]

Step 7: Results Summary
==================================================================

Simulation Results Summary
==================================================
Duration: 45.2 seconds
Flights: 93
Total Revenue: $1,547,230.00
Total Bookings: 4,823
Cancellations: 241 (5.0%)
Load Factor: 73.5%
Average Fare: $320.68
RM Optimizations: 31

==================================================================
Simulation Complete!
==================================================================
```

## Core Concepts

### 1. Flight Schedules

Define your airline's schedule:

```python
from core.models import FlightSchedule, Aircraft, Route

schedule = FlightSchedule(
    airline_code="AA",
    flight_number="100",
    route=jfk_to_lax_route,
    departure_time=time(8, 0),
    arrival_time=time(11, 30),
    days_of_week=[0, 1, 2, 3, 4, 5, 6],  # Daily
    aircraft=boeing_777,
    valid_from=date(2025, 1, 1),
    valid_until=date(2025, 12, 31)
)
```

### 2. Demand Generation

Configure realistic passenger demand:

```python
from demand.generator import DemandStreamConfig

demand = DemandStreamConfig(
    stream_id="JFK-LAX",
    origin=jfk,
    destination=lax,
    mean_daily_demand=150.0,
    business_proportion=0.35,  # 35% business travelers
    business_wtp_mean=900.0,   # Average willingness-to-pay
    leisure_wtp_mean=350.0
)
```

### 3. Revenue Management

Configure RM optimization:

```python
from core.simulator import SimulationConfig

config = SimulationConfig(
    start_date=date(2025, 1, 1),
    end_date=date(2025, 3, 31),
    rm_method="EMSR-b",  # or "DP", "MC"
    optimization_horizons=[30, 14, 7, 3, 1],  # Days before departure
    dynamic_pricing=True
)
```

### 4. Running the Simulation

```python
from core.simulator import Simulator

# Create simulator
simulator = Simulator(
    config=config,
    schedules=schedules,
    routes=routes,
    airports=airports
)

# Generate and add demand
requests = demand_generator.generate_all_requests(
    start_date=config.start_date,
    end_date=config.end_date
)
demand_generator.add_requests_to_event_queue(
    simulator.event_manager,
    requests
)

# Run simulation
results = simulator.run()

# Analyze results
print(f"Total Revenue: ${results.total_revenue:,.2f}")
print(f"Load Factor: {results.load_factor:.1%}")
```

## Key Features

### Multi-Airline Competition

Simulate competitive markets with multiple airlines:

```python
# Coming soon in competitive simulation module
from competition import Airline, CompetitiveSimulator

airlines = [
    Airline("AA", strategy="aggressive", capacity=150),
    Airline("UA", strategy="conservative", capacity=180),
    Airline("DL", strategy="ml_based", capacity=160)
]

sim = CompetitiveSimulator(airlines=airlines)
results = sim.run()
```

### Dynamic Pricing

Enable real-time price adjustments:

```python
config = SimulationConfig(
    dynamic_pricing=True,
    price_update_frequency_hours=6.0
)
```

### Advanced RM Algorithms

Choose from multiple optimization methods:

- **EMSR-b**: Industry standard, fast and accurate
- **EMSR-a**: Simpler variant
- **Dynamic Programming**: Theoretically optimal
- **Monte Carlo**: Simulation-based
- **ML-Enhanced**: Machine learning (coming soon)

```python
from rm.optimizer import RMOptimizer

optimizer = RMOptimizer(method="EMSR-b")
control = optimizer.optimize(flight_date, forecasts, fares)
```

## Project Structure

```
pyairline_rm/
├── core/                    # Core simulation engine
│   ├── models.py           # Data models
│   ├── events.py           # Event system
│   └── simulator.py        # Main simulator
├── demand/                  # Demand generation
│   ├── generator.py        # Request generation
│   └── forecaster.py       # Demand forecasting
├── rm/                      # Revenue management
│   ├── optimizer.py        # RM algorithms
│   └── emsr.py             # EMSR implementations
├── pricing/                 # Fare management
│   └── fare_engine.py      # Pricing logic
├── inventory/               # Inventory control
│   └── manager.py          # Availability management
├── choice/                  # Customer choice
│   └── model.py            # Choice modeling
├── competition/             # Multi-airline
│   └── airline.py          # Airline agents
├── analytics/               # Analysis
│   └── metrics.py          # Performance metrics
└── examples/                # Example scripts
    └── basic_example.py    # Basic simulation
```

## Next Steps

### 1. Customize the Network

Edit `basic_example.py` to add your own:
- Airports
- Routes
- Flight schedules
- Demand patterns

### 2. Experiment with RM Methods

Try different optimization algorithms:
```python
config = SimulationConfig(
    rm_method="DP"  # Try Dynamic Programming
)
```

### 3. Add Multiple Airlines

Simulate competitive markets (module in development).

### 4. Analyze Results

Export results for analysis:
```python
import pandas as pd

# Convert bookings to DataFrame
df = pd.DataFrame([
    {
        'booking_id': b.booking_id,
        'revenue': b.total_revenue,
        'party_size': b.party_size,
        'flight': b.solution.flights[0].schedule.flight_code
    }
    for b in results.bookings
])

df.to_csv('booking_results.csv')
```

### 5. Visualize Performance

Create visualizations:
```python
import plotly.express as px

# Load factor by flight
load_factors = [
    {
        'flight': fd.schedule.flight_code,
        'date': fd.departure_date,
        'load_factor': fd.load_factor()
    }
    for fd in simulator.flight_dates.values()
]

df_lf = pd.DataFrame(load_factors)
fig = px.line(df_lf, x='date', y='load_factor', color='flight',
              title='Load Factor Evolution')
fig.show()
```

## Troubleshooting

### Issue: Import Errors

**Solution**: Ensure you're in the virtual environment and installed dependencies:
```bash
source pyairline_env/bin/activate
pip install -r requirements.txt
```

### Issue: No Demand Generated

**Solution**: Check that departure dates fall within simulation period:
```python
# Ensure sufficient advance booking window
demand_config = DemandStreamConfig(
    mean_advance_purchase=21.0,
    max_advance_days=90  # Allow bookings up to 90 days out
)
```

### Issue: Low Booking Success Rate

**Solution**: 
1. Check WTP distributions match fare levels
2. Increase demand volume
3. Verify capacity is sufficient

### Issue: Slow Simulation

**Solution**:
1. Reduce simulation period
2. Use EMSR-b instead of DP
3. Enable parallel processing (coming soon)

## Advanced Configuration

### Custom Booking Curves

Define how demand varies by days to departure:

```python
booking_curve = {
    0: 0.1,   # Very low last minute
    7: 0.8,   # Rising week before
    14: 1.2,  # Peak 2 weeks out
    30: 1.0,  # Normal 1 month out
    60: 0.6,  # Lower far out
}

demand_config = DemandStreamConfig(
    booking_curve=booking_curve,
    # ... other params
)
```

### Seasonality

Add seasonal demand patterns:

```python
seasonality = {
    1: 0.8,   # January (low)
    7: 1.3,   # July (peak summer)
    12: 1.2,  # December (holidays)
}

demand_config = DemandStreamConfig(
    seasonality=seasonality,
    # ... other params
)
```

### Day of Week Patterns

Model different demand by weekday:

```python
dow_pattern = {
    0: 1.2,  # Monday (high business)
    4: 1.3,  # Friday (high)
    6: 0.7,  # Sunday (low)
}

demand_config = DemandStreamConfig(
    day_of_week_pattern=dow_pattern,
    # ... other params
)
```

## Support and Resources

- **Documentation**: See `docs/` directory
- **Examples**: Check `examples/` directory
- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions

## Contributing

Contributions welcome! See `CONTRIBUTING.md` for guidelines.

## License

MIT License - see `LICENSE` file for details.
