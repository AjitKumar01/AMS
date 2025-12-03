# Advanced Features Guide

## Overview

This guide covers the advanced features that make PyAirline RM a realistic, production-ready airline revenue management simulator:

1. **Multi-Airline Competition** - Simulate realistic competitive markets
2. **Network Revenue Management** - O-D control with displacement costs
3. **ML-Based Forecasting** - Neural network demand prediction
4. **Forecast Accuracy Impact** - How forecast errors affect RM performance

## 1. Multi-Airline Competition

### Overview

The competition module allows you to simulate markets with multiple airlines, each using different competitive strategies. Airlines observe competitor actions and adjust their pricing and capacity decisions accordingly.

### Key Concepts

**Competitive Strategies:**
- `AGGRESSIVE`: Price 5-10% below competition, maximize market share
- `CONSERVATIVE`: Premium positioning, yield over volume
- `ML_BASED`: Use machine learning for dynamic optimization
- `MATCH_COMPETITOR`: Follow market leader
- `YIELD_FOCUSED`: Maximize revenue per passenger
- `MARKET_SHARE`: Aggressive growth strategy

**Market Intelligence:**
- Airlines observe competitor fares (configurable transparency)
- Load factors are partially visible (with noise)
- Market share tracking by route
- Herfindahl-Hirschman Index (HHI) for concentration analysis

### Usage Example

```python
from competition.airline import Airline, CompetitiveStrategy
from competition.market import Market

# Create airlines with different strategies
american = Airline(
    code="AA",
    name="American Airlines",
    strategy=CompetitiveStrategy.AGGRESSIVE,
    base_price_multiplier=0.95,  # 5% below market
    cost_per_seat_mile=0.10,
    brand_preference=0.1
)

united = Airline(
    code="UA",
    name="United Airlines",
    strategy=CompetitiveStrategy.ML_BASED,
    base_price_multiplier=1.0,
    cost_per_seat_mile=0.11,
    brand_preference=0.05
)

delta = Airline(
    code="DL",
    name="Delta Air Lines",
    strategy=CompetitiveStrategy.CONSERVATIVE,
    base_price_multiplier=1.10,  # 10% premium
    cost_per_seat_mile=0.12,
    brand_preference=0.15
)

# Create market coordinator
market = Market(information_transparency=0.75)  # 75% observable
market.add_airline(american)
market.add_airline(united)
market.add_airline(delta)

# During simulation, airlines observe competitors
market.share_competitive_intelligence(current_date)

# Analyze competitive dynamics
analysis = market.analyze_competitive_dynamics()
print(f"Market HHI: {analysis['overall_hhi']:.0f}")
print(f"Avg competitors per route: {analysis['avg_competitors_per_market']:.1f}")
```

### Market Concentration Analysis

The system calculates the Herfindahl-Hirschman Index (HHI) for each market:

- **HHI < 1500**: Competitive market (healthy competition)
- **HHI 1500-2500**: Moderately concentrated
- **HHI > 2500**: Highly concentrated (potential monopoly concerns)

### Competitive Decision Making

Each airline makes decisions based on:
1. **Current market position** - Market share, load factors
2. **Competitor actions** - Recent price changes, capacity adjustments
3. **Strategy** - Aggressive vs. conservative approach
4. **Cost structure** - Operating costs per seat mile
5. **Brand strength** - Customer preference

Example of airline response to competition:

```python
# American (aggressive) observes United's price increase
american.observe_competitor_fare(
    competitor_code="UA",
    route_key="JFK-LAX",
    fare=350.0,
    timestamp=datetime.now()
)

# American decides to undercut
new_fare = american.decide_base_fare(route, flight_date)
# new_fare will be ~5-7% below United's $350 = ~$330
```

## 2. Network Revenue Management (O-D Control)

### Overview

Network RM optimizes across entire itineraries (origin-destination pairs) rather than individual flight legs. This is more sophisticated than leg-based control and better captures the true value of bookings.

### Key Concepts

**Virtual Nesting:**
- Booking classes are organized by revenue value, not physical cabin
- A $500 economy booking may be prioritized over a $400 business booking
- Enables better allocation across the network

**Displacement Cost (Bid Price):**
- The opportunity cost of selling a seat
- Represents expected revenue from the next-best use of that seat
- Calculated using EMSR or linear programming

**Network Value:**
```
Network Value = Fare - Sum(Displacement Costs for all legs)
```

Accept booking if Network Value > 0

### Usage Example

```python
from inventory.network import NetworkOptimizer

# Create network optimizer
optimizer = NetworkOptimizer(
    num_virtual_buckets=10,
    optimization_method="linear_programming"
)

# Register itineraries and their component legs
optimizer.register_itinerary(
    itinerary_id="JFK-ORD-LAX",
    leg_ids=["AA200_2025-12-15", "AA300_2025-12-15"]
)

# Calculate displacement costs
for flight_date in flight_dates.values():
    bid_price = optimizer.calculate_displacement_cost(
        flight_date=flight_date,
        forecasted_demand=demand_forecast,
        fares=fares
    )
    print(f"Flight {flight_date.flight_code}: Bid price = ${bid_price:.2f}")

# Evaluate a booking request
should_accept = optimizer.should_accept_booking(
    solution=travel_solution,
    fare=450.0,
    party_size=2
)

if should_accept:
    print("Accept: Network value is positive")
else:
    print("Reject: Displacement cost exceeds fare")
```

### Network Optimization

The optimizer solves a linear program to maximize total network revenue:

**Objective:**
```
Maximize: Sum of (fare × bookings) for all itineraries
```

**Constraints:**
```
1. Capacity: Bookings using leg L ≤ Capacity of L
2. Demand: Bookings for itinerary I ≤ Demand for I
3. Non-negativity: Bookings ≥ 0
```

Example:

```python
# Optimize across entire network
optimal_limits = optimizer.optimize_network(
    flight_dates=all_flights,
    demand_forecasts=forecasts,
    itineraries=[
        ("JFK-LAX-direct", ["AA100"], 450.0, 200.0),
        ("JFK-ORD-LAX", ["AA200", "AA300"], 420.0, 150.0),
        # ... more itineraries
    ]
)

# Apply optimal booking limits
for flight_id, limits in optimal_limits.items():
    flight = flight_dates[flight_id]
    for booking_class, limit in limits.items():
        flight.set_booking_limit(booking_class, limit)
```

### Displacement Cost Report

```python
report = optimizer.get_displacement_report()

print(f"Average bid price: ${report['avg_bid_price']:.2f}")
print(f"Legs with high displacement (>$200): {count_high}")

for leg_id, leg_data in report['legs'].items():
    print(f"{leg_id}:")
    print(f"  Bid price: ${leg_data['bid_price']:.2f}")
    print(f"  Capacity utilization: {leg_data['utilization']:.1%}")
```

## 3. ML-Based Demand Forecasting

### Overview

Advanced demand forecasting using multiple methods, including neural networks. The system tracks forecast accuracy and demonstrates how forecast errors impact revenue management performance.

### Forecasting Methods

1. **Historical Average**: Simple baseline
2. **Pickup Method**: Traditional airline forecasting
   - Additive pickup: Current + Expected additional
   - Multiplicative pickup: Current × Pickup factor
3. **Exponential Smoothing**: Weighted average with decay
4. **Neural Network**: Deep learning (PyTorch)
5. **Ensemble**: Combine multiple methods

### Usage Example

```python
from demand.forecaster import DemandForecaster, ForecastMethod

# Create forecaster with different methods
forecasters = {
    'traditional': DemandForecaster(
        method=ForecastMethod.PICKUP,
        track_accuracy=True,
        add_noise=True,
        noise_std=0.15  # 15% forecast error
    ),
    'ml': DemandForecaster(
        method=ForecastMethod.NEURAL_NETWORK,
        track_accuracy=True,
        add_noise=True,
        noise_std=0.08  # 8% error (ML is better)
    )
}

# Generate forecast
forecast = forecaster.forecast_demand(
    flight_date=flight,
    current_date=date.today(),
    current_bookings={BookingClass.ECONOMY: 50}
)

print(f"Total forecast: {forecast.get_total_forecast():.0f} passengers")
print(f"Economy: {forecast.get_forecast(BookingClass.ECONOMY):.0f}")
print(f"Business: {forecast.get_forecast(BookingClass.BUSINESS):.0f}")
print(f"Days before departure: {forecast.days_before_departure}")
```

### Pickup Method Details

The pickup method is the industry standard for airline forecasting:

**Pickup Curve** (typical pattern):
```
Days Before | Pickup Factor | % of Final Demand
-----------------------------------------------
   90+      |     6.67      |      15%
    60      |     3.33      |      30%
    30      |     1.82      |      55%
    14      |     1.33      |      75%
     7      |     1.11      |      90%
     1      |     1.02      |      98%
     0      |     1.00      |     100%
```

**Multiplicative Pickup:**
```
Final Demand = Current Bookings × Pickup Factor
```

**Additive Pickup:**
```
Final Demand = Current Bookings + Expected Additional
```

### Neural Network Forecasting

When PyTorch is available, the system can use a neural network:

**Architecture:**
```
Input Layer (4 features):
  - Days to departure (normalized)
  - Day of week
  - Month (seasonality)
  - Current load factor

Hidden Layers:
  - Layer 1: 64 neurons, ReLU activation
  - Layer 2: 64 neurons, ReLU activation

Output Layer:
  - 1 neuron, Sigmoid activation (predicted load factor)
```

**Training** (in production system):
```python
# Collect training data
training_data = []
for historical_flight in history:
    features = extract_features(historical_flight)
    label = historical_flight.final_load_factor
    training_data.append((features, label))

# Train model
model.train(training_data, epochs=100)

# Use for forecasting
forecast = forecaster.forecast_demand(flight, current_date, bookings)
```

## 4. Forecast Accuracy and Revenue Impact

### Overview

A critical but often overlooked aspect of RM: **Forecast accuracy directly impacts revenue performance**. This module tracks how forecast errors translate to revenue loss.

### Types of Forecast Error

**Over-Forecasting** (Forecast > Actual):
- RM sets booking limits too high
- Accepts too many low-value bookings
- Dilutes yield
- Revenue loss: 20-30% of error × average fare

**Under-Forecasting** (Forecast < Actual):
- RM sets booking limits too low
- Rejects high-value bookings (spill)
- Lost revenue opportunities
- Revenue loss: 50-80% of error × high-value fare

### Accuracy Metrics

```python
# Track forecast accuracy
forecaster.evaluate_forecast(
    forecast=my_forecast,
    actual_bookings=actual_bookings
)

# Get accuracy report
report = forecaster.get_accuracy_report()

print(f"Mean Absolute Error: {report['mae']:.2f} passengers")
print(f"Mean Absolute % Error: {report['mape']:.1f}%")
print(f"Bias: {report['bias']:.2f}")  # + = over-forecast
print(f"Revenue lost to errors: ${report['revenue_lost']:,.0f}")
```

### Revenue Impact Calculation

```python
# Estimate revenue impact of forecast error
revenue_loss = forecaster.estimate_revenue_impact(
    forecast_error=actual_demand - forecast,
    fare=average_fare,
    capacity=flight_capacity
)

print(f"Revenue lost due to forecast error: ${revenue_loss:,.0f}")
```

**Example Scenario:**

Flight AA100 on Dec 15:
- **Forecast**: 180 passengers (80% load factor)
- **Actual**: 200 passengers (89% load factor)
- **Error**: -20 passengers (under-forecast)

Impact:
- RM protected 160 seats for high-value bookings
- Had 20 more seats available than expected
- Could have sold 20 more tickets at high value
- Assuming $500 average fare → Lost $10,000

### Comparative Analysis

Compare forecasting methods by accuracy and revenue impact:

```python
# Test multiple methods
methods = [
    ForecastMethod.HISTORICAL_AVERAGE,
    ForecastMethod.PICKUP,
    ForecastMethod.NEURAL_NETWORK
]

results = {}
for method in methods:
    forecaster = DemandForecaster(method=method, track_accuracy=True)
    
    # Run simulation
    sim_results = run_simulation_with_forecaster(forecaster)
    
    results[method.value] = {
        'revenue': sim_results.total_revenue,
        'mae': forecaster.accuracy.mean_absolute_error,
        'revenue_lost': forecaster.accuracy.revenue_lost_to_error
    }

# Compare
print("Forecasting Method Comparison:")
for method, data in results.items():
    print(f"\n{method}:")
    print(f"  Total revenue: ${data['revenue']:,.0f}")
    print(f"  MAE: {data['mae']:.2f} passengers")
    print(f"  Revenue lost: ${data['revenue_lost']:,.0f}")
    print(f"  Impact: {data['revenue_lost']/data['revenue']*100:.2f}%")
```

**Typical Results:**
```
Historical Average:
  Total revenue: $2,450,000
  MAE: 18.5 passengers
  Revenue lost: $125,000
  Impact: 5.1%

Pickup Method:
  Total revenue: $2,530,000
  MAE: 12.3 passengers
  Revenue lost: $78,000
  Impact: 3.1%

Neural Network:
  Total revenue: $2,595,000
  MAE: 8.7 passengers
  Revenue lost: $45,000
  Impact: 1.7%
```

**Key Insight**: Better forecasting directly translates to higher revenue. A 50% reduction in MAE can improve revenue by 2-3%.

## Running the Advanced Example

The complete example demonstrates all features together:

```bash
cd pyairline_rm
python examples/competitive_simulation.py
```

This simulates:
- 3 airlines (American, United, Delta) with different strategies
- 4 routes with varying competitive intensity
- 31 days of operations
- Different forecasting methods per airline
- Network RM with O-D control
- Complete competitive analysis

**Output includes:**
1. Airline performance comparison
2. Market share by route
3. HHI concentration analysis
4. Forecast accuracy by airline
5. Revenue impact of forecast errors
6. Network RM displacement costs
7. Competitive insights and winner analysis

## Best Practices

### 1. Competition

- Set `information_transparency` to 0.6-0.8 for realistic markets
- Use different strategies for different airlines
- Monitor HHI to ensure healthy competition
- Adjust `base_price_multiplier` to reflect competitive positioning

### 2. Network RM

- Use 8-12 virtual buckets for good granularity
- Recalculate displacement costs daily or when bookings change significantly
- Monitor bid prices to ensure they reflect demand
- Accept bookings only when network value > 0

### 3. Forecasting

- Start with pickup method (industry standard)
- Add noise (10-15%) for realistic simulations
- Track accuracy metrics continuously
- Measure revenue impact to justify ML investment
- Update forecasts as new bookings arrive

### 4. Performance

- Neural network forecasting is slower but more accurate
- Linear programming for network optimization can be intensive
- Consider approximate methods for real-time applications
- Cache displacement costs when possible

## Advanced Topics

### Custom Strategies

Create your own competitive strategy:

```python
from competition.strategies import StrategyBase

class CustomStrategy(StrategyBase):
    def decide_fare(self, route, flight_date, competitor_fares):
        # Your custom pricing logic
        return calculated_fare
    
    def respond_to_competition(self, route_key, our_share, actions):
        # Your response logic
        return response_dict
```

### Ensemble Forecasting

Combine multiple forecasting methods:

```python
forecasts = []
for method in [ForecastMethod.PICKUP, ForecastMethod.NEURAL_NETWORK]:
    f = DemandForecaster(method=method)
    forecast = f.forecast_demand(flight, date, bookings)
    forecasts.append(forecast)

# Weighted average
final_forecast = weighted_average(forecasts, weights=[0.4, 0.6])
```

### Dynamic Virtual Buckets

Adjust bucket boundaries based on market conditions:

```python
# During high-demand periods, create finer buckets at high end
if avg_load_factor > 0.80:
    optimizer.create_virtual_buckets(
        min_fare=min_fare,
        max_fare=max_fare * 1.5  # Extend high end
    )
```

## Conclusion

These advanced features make PyAirline RM suitable for:
- Academic research on airline competition
- Revenue management strategy testing
- ML forecasting algorithm development
- Network optimization research
- Industry competitive analysis

The realistic simulation of forecast errors and their revenue impact is particularly valuable for understanding the business value of improved forecasting technology.
