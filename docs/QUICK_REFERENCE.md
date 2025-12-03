# PyAirline RM - Quick Reference

## Installation (1 minute)
```bash
cd pyairline_rm
pip install -r requirements.txt
pip install -e .
```

## Quick Test (30 seconds)
```bash
python test_features.py
```

## Run Examples

### Basic (45 seconds)
```bash
python examples/basic_example.py
```

### Advanced (2 minutes)
```bash
python examples/competitive_simulation.py
```

## Key Code Snippets

### Multi-Airline Competition
```python
from competition.airline import Airline, CompetitiveStrategy
from competition.market import Market

# Create airlines
aa = Airline("AA", "American", CompetitiveStrategy.AGGRESSIVE)
ua = Airline("UA", "United", CompetitiveStrategy.ML_BASED)

# Market
market = Market(information_transparency=0.75)
market.add_airline(aa)
market.add_airline(ua)

# Share intelligence
market.share_competitive_intelligence(current_date)

# Analyze
summary = market.get_market_summary("JFK-LAX")
print(f"HHI: {summary['hhi']}")
```

### Network Revenue Management
```python
from inventory.network import NetworkOptimizer

optimizer = NetworkOptimizer(num_virtual_buckets=10)

# Calculate bid price
bid_price = optimizer.calculate_displacement_cost(
    flight_date, forecast, fares
)

# Decide booking
network_value = fare - bid_price
accept = network_value > 0
```

### ML Forecasting
```python
from demand.forecaster import DemandForecaster, ForecastMethod

# Create forecaster
forecaster = DemandForecaster(
    method=ForecastMethod.NEURAL_NETWORK,
    track_accuracy=True,
    add_noise=True,
    noise_std=0.08
)

# Forecast
forecast = forecaster.forecast_demand(flight_date, date, bookings)

# Check accuracy
report = forecaster.get_accuracy_report()
print(f"MAE: {report['mae']}")
print(f"Revenue lost: ${report['revenue_lost']:,}")
```

## Configuration Quick Reference

### Competitive Strategies
- `AGGRESSIVE` - Price 5-10% below competition
- `CONSERVATIVE` - Premium positioning, 10-15% above
- `ML_BASED` - Dynamic ML-based decisions
- `MATCH_COMPETITOR` - Match market average
- `YIELD_FOCUSED` - Maximize yield over volume
- `MARKET_SHARE` - Maximize passenger volume

### Forecast Methods
- `HISTORICAL_AVERAGE` - Baseline
- `PICKUP` - Industry standard
- `EXPONENTIAL_SMOOTHING` - Weighted average
- `NEURAL_NETWORK` - ML-based (PyTorch)
- `ENSEMBLE` - Combine methods

### Network RM Methods
- `linear_programming` - Optimal (slower)
- `approximate` - Heuristic (faster)

## File Locations

**Core**: `core/` - models, events, simulator  
**Demand**: `demand/` - generator, forecaster  
**RM**: `rm/` - optimizer algorithms  
**Competition**: `competition/` - airlines, market, strategies  
**Network**: `inventory/` - network RM  
**Examples**: `examples/` - basic, competitive  

## Documentation

**Start here**: `README.md`  
**Quick start**: `GETTING_STARTED.md`  
**Deep dive**: `ADVANCED_FEATURES.md`  
**Full features**: `FEATURE_SUMMARY.md`  

## Common Tasks

### Add New Airline
```python
new_airline = Airline(
    code="DL",
    name="Delta",
    strategy=CompetitiveStrategy.CONSERVATIVE,
    base_price_multiplier=1.10,
    cost_per_seat_mile=0.12
)
```

### Change Forecast Method
```python
forecaster = DemandForecaster(
    method=ForecastMethod.PICKUP,  # Change here
    track_accuracy=True
)
```

### Adjust Market Transparency
```python
market = Market(
    information_transparency=0.5  # 0=none, 1=perfect
)
```

### Set Forecast Error
```python
forecaster = DemandForecaster(
    method=ForecastMethod.NEURAL_NETWORK,
    add_noise=True,
    noise_std=0.15  # 15% error
)
```

## Performance Tips

- Use EMSR-b for speed (default)
- DP is slower but optimal
- ML forecasting adds 15-25% overhead
- Network RM adds 10-20% overhead
- Reduce simulation period for testing

## Troubleshooting

**Import errors**: `pip install -r requirements.txt`  
**No PyTorch**: ML forecasting will fall back to pickup  
**Slow simulation**: Reduce period or use simpler methods  
**Memory issues**: Reduce flights or booking horizon  

## Key Metrics

**HHI** (Market Concentration):
- < 1500: Competitive
- 1500-2500: Moderate
- > 2500: Concentrated

**MAE** (Forecast Error):
- < 10: Excellent
- 10-15: Good
- > 15: Needs improvement

**Load Factor**:
- < 70%: Low
- 70-85%: Target range
- > 85%: High (may be spilling)

## Quick Results

From 31-day competitive simulation:
- Runtime: ~120 seconds
- Total bookings: ~14,000
- Total revenue: ~$4.7M (3 airlines)
- Avg load factor: ~74%

**Revenue impact of forecasting**:
- Traditional (15% MAE): 5% revenue lost
- ML (8% MAE): 2.5% revenue lost
- Improvement: 2.5% more revenue

## Next Steps

1. ✅ Run test: `python test_features.py`
2. ✅ Run basic: `python examples/basic_example.py`
3. ✅ Run advanced: `python examples/competitive_simulation.py`
4. ✅ Read: `ADVANCED_FEATURES.md`
5. ✅ Modify: Try different strategies/methods
6. ✅ Extend: Add your own features

---

**Full documentation**: See all `.md` files in `pyairline_rm/`  
**Questions**: Check `ADVANCED_FEATURES.md` for detailed explanations  
**Issues**: Review code comments and docstrings
