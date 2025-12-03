# PyAirline RM - Advanced Airline Revenue Management Simulator

A modern, Python-based airline revenue management simulator designed for research, education, and competitive analysis.

## Key Features

### Core Capabilities
- **Multi-Airline Competition**: Simulate multiple competing airlines with different strategies
- **Network Revenue Management**: O-D based control with virtual nesting
- **Dynamic Pricing**: Real-time fare adjustments based on demand and competition
- **Advanced Forecasting**: ML-enhanced demand prediction with pickup methods
- **Customer Segmentation**: Business/leisure with realistic booking patterns
- **Real-time Optimization**: Multiple RM algorithms (EMSR-b, Dynamic Programming, ML-based)

### Improvements Over Existing C++ System

1. **Modern Python Stack**
   - NumPy/SciPy for numerical optimization
   - Pandas for data management and analysis
   - Scikit-learn for ML-based forecasting
   - Plotly/Dash for interactive visualization

2. **Enhanced Features**
   - **Competitive Dynamics**: Airlines react to competitor pricing and capacity
   - **Dynamic Pricing**: Continuous price optimization (not just discrete classes)
   - **Machine Learning**: Neural network-based demand forecasting
   - **Network Effects**: True O-D control with displacement costs
   - **Real-time Analytics**: Live dashboard during simulation
   - **A/B Testing Framework**: Compare RM strategies side-by-side

3. **Better User Experience**
   - Configuration via YAML/JSON
   - Interactive web dashboard
   - Comprehensive visualization
   - Export to various formats (CSV, Excel, Parquet)
   - RESTful API for integration

4. **Advanced Modeling**
   - Price elasticity of demand
   - Competitor response functions
   - Customer loyalty and repeat booking
   - Ancillary revenue (baggage, seats, etc.)
   - Group bookings and corporate contracts
   - Codeshare and alliance considerations

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run a basic simulation
python -m pyairline_rm.main --config examples/basic_competition.yaml

# Launch interactive dashboard
python -m pyairline_rm.dashboard

# Run benchmark scenarios
python -m pyairline_rm.benchmarks
```

## Project Structure

```
pyairline_rm/
├── core/               # Core business logic
│   ├── models.py      # Data models (Flight, Booking, etc.)
│   ├── events.py      # Event system
│   └── simulator.py   # Main simulation engine
├── demand/            # Demand generation
│   ├── generator.py   # Demand stream generation
│   ├── forecaster.py  # Demand forecasting
│   └── customer.py    # Customer behavior models
├── inventory/         # Inventory management
│   ├── manager.py     # Inventory control
│   ├── availability.py # Availability calculation
│   └── network.py     # Network O-D management
├── pricing/           # Fare management
│   ├── fare_engine.py # Fare calculation
│   ├── dynamic.py     # Dynamic pricing
│   └── rules.py       # Fare rules engine
├── rm/                # Revenue management
│   ├── optimizer.py   # RM optimization
│   ├── emsr.py        # EMSR algorithms
│   ├── dynamic_prog.py # Dynamic programming
│   └── ml_optimizer.py # ML-based optimization
├── competition/       # Multi-airline competition
│   ├── airline.py     # Airline agent
│   ├── strategy.py    # Competitive strategies
│   └── market.py      # Market dynamics
├── choice/            # Customer choice
│   ├── model.py       # Choice modeling
│   └── utility.py     # Utility functions
├── analytics/         # Analysis and reporting
│   ├── metrics.py     # Performance metrics
│   ├── visualizer.py  # Visualization
│   └── exporter.py    # Data export
├── dashboard/         # Web dashboard
│   ├── app.py         # Dash application
│   └── components.py  # UI components
└── utils/             # Utilities
    ├── config.py      # Configuration management
    └── helpers.py     # Helper functions
```

## Usage Examples

### Basic Single-Airline Simulation

```python
from pyairline_rm import Simulator
from pyairline_rm.config import load_config

# Load configuration
config = load_config('examples/single_airline.yaml')

# Create and run simulation
sim = Simulator(config)
results = sim.run()

# Analyze results
print(f"Total Revenue: ${results.total_revenue:,.2f}")
print(f"Load Factor: {results.load_factor:.1%}")
print(f"Average Fare: ${results.average_fare:.2f}")
```

### Competitive Simulation with Network RM

```python
from competition.airline import Airline, CompetitiveStrategy
from competition.market import Market
from inventory.network import NetworkOptimizer
from demand.forecaster import DemandForecaster, ForecastMethod

# Create competing airlines
american = Airline(
    code="AA",
    name="American Airlines",
    strategy=CompetitiveStrategy.AGGRESSIVE,
    base_price_multiplier=0.95  # Price 5% below market
)

united = Airline(
    code="UA",
    name="United Airlines",
    strategy=CompetitiveStrategy.ML_BASED,
    base_price_multiplier=1.0
)

# Market coordinator
market = Market(information_transparency=0.75)
market.add_airline(american)
market.add_airline(united)

# Network optimizer
network_optimizer = NetworkOptimizer(
    num_virtual_buckets=10,
    optimization_method="linear_programming"
)

# ML-based forecaster
forecaster = DemandForecaster(
    method=ForecastMethod.NEURAL_NETWORK,
    track_accuracy=True,
    add_noise=True,
    noise_std=0.08  # 8% forecast error
)

# Generate forecast
forecast = forecaster.forecast_demand(flight_date, current_date, bookings)

# Calculate displacement cost
bid_price = network_optimizer.calculate_displacement_cost(
    flight_date, forecast.forecasts, fares
)

# Evaluate booking based on network value
network_value = fare - bid_price
accept_booking = network_value > 0

# Share competitive intelligence
market.share_competitive_intelligence(current_date)

# Analyze market
summary = market.get_market_summary("JFK-LAX")
print(f"Market HHI: {summary['hhi']:.0f}")
print(f"AA Market Share: {summary['airlines']['AA']['market_share_pax']:.1%}")

# Check forecast accuracy
accuracy = forecaster.get_accuracy_report()
print(f"Forecast MAE: {accuracy['mae']:.2f} passengers")
print(f"Revenue lost to errors: ${accuracy['revenue_lost']:,.0f}")
```

### Multi-Airline Competition

```python
from pyairline_rm import CompetitiveSimulator
from pyairline_rm.competition import Airline, AggressivePricing, ConservativeRM

# Define airlines
airlines = [
    Airline("AA", strategy=AggressivePricing(), capacity=150),
    Airline("UA", strategy=ConservativeRM(), capacity=180),
    Airline("BA", strategy=MLBasedOptimization(), capacity=160)
]

# Run competitive simulation
sim = CompetitiveSimulator(airlines=airlines, duration_days=90)
results = sim.run()

# Compare performance
for airline in airlines:
    print(f"{airline.code}: Rev=${results[airline.code].revenue:,.0f}, "
          f"LF={results[airline.code].load_factor:.1%}")
```

### Dynamic Pricing Optimization

```python
from pyairline_rm.pricing import DynamicPricingEngine

# Configure dynamic pricing
pricing = DynamicPricingEngine(
    base_fare=300,
    min_fare=150,
    max_fare=800,
    elasticity=-1.5,
    update_frequency='hourly'
)

# Run with dynamic pricing
sim = Simulator(config, pricing_engine=pricing)
results = sim.run()

# Analyze pricing evolution
pricing.plot_price_evolution()
```

## Configuration

Example YAML configuration:

```yaml
simulation:
  start_date: "2025-01-01"
  end_date: "2025-03-31"
  seed: 42

airlines:
  - code: "AA"
    name: "American Airlines"
    strategy: "aggressive"
    capacity: 150
  
  - code: "UA"
    name: "United Airlines"
    strategy: "conservative"
    capacity: 180

routes:
  - origin: "JFK"
    destination: "LAX"
    distance: 2475
    daily_frequency: 5
    
demand:
  mean_daily: 500
  seasonality: true
  segments:
    business:
      proportion: 0.30
      wtp_mean: 800
      wtp_std: 200
    leisure:
      proportion: 0.70
      wtp_mean: 300
      wtp_std: 100

revenue_management:
  method: "emsr_b"
  optimization_frequency: "daily"
  forecasting: "ml_enhanced"
  overbooking: true
  
pricing:
  mode: "dynamic"
  classes: ["Y", "B", "M", "Q"]
  dynamic_adjustment: true
```

## Advanced Features

### Network Revenue Management

```python
from pyairline_rm.inventory import NetworkRMOptimizer

# Define network with connecting flights
network = NetworkRMOptimizer()
network.add_route("JFK", "LAX", direct=True)
network.add_route("JFK", "SFO", via="DEN")
network.add_route("JFK", "SEA", via="ORD")

# Optimize considering network effects
network.optimize_network(method='bid_price')
```

### Machine Learning Integration

```python
from pyairline_rm.rm import MLOptimizer

# Train ML model on historical data
ml_optimizer = MLOptimizer()
ml_optimizer.train(historical_data)

# Use for demand forecasting
forecast = ml_optimizer.predict(
    dtd=30,
    bookings_sofar=45,
    day_of_week='Monday',
    seasonality_factor=1.2
)
```

### Real-time Dashboard

```python
from pyairline_rm.dashboard import launch_dashboard

# Launch interactive dashboard
launch_dashboard(
    sim=simulator,
    port=8050,
    auto_refresh=True
)
# Navigate to http://localhost:8050
```

## Performance Benchmarks

Typical simulation performance on standard hardware:

- **Small** (10 flights, 30 days, 10K bookings): < 1 minute
- **Medium** (100 flights, 90 days, 100K bookings): ~5 minutes
- **Large** (1000 flights, 365 days, 1M bookings): ~30 minutes
- **Network** (50 O-D pairs, 180 days, 500K bookings): ~15 minutes

## Research Applications

- Compare RM algorithms (EMSR vs. DP vs. ML)
- Study competitive dynamics and Nash equilibria
- Analyze dynamic pricing vs. fixed fare classes
- Test new forecasting methods
- Validate theoretical models with realistic demand
- A/B test operational strategies

## License

MIT License - See LICENSE file

## Contributing

Contributions welcome! See CONTRIBUTING.md

## Citation

If you use this simulator in research, please cite:

```bibtex
@software{pyairline_rm,
  title={PyAirline RM: Advanced Airline Revenue Management Simulator},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/pyairline_rm}
}
```

## Support

- Documentation: [docs/](docs/)
- Issues: [GitHub Issues](https://github.com/yourusername/pyairline_rm/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/pyairline_rm/discussions)
