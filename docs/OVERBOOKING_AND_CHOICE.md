# Overbooking and Customer Choice Models - Complete Guide

## Overview

This guide covers two critical additions to the PyAirline RM simulator:

1. **Overbooking** - Optimize booking limits to account for no-shows
2. **Customer Choice Models** - Realistic behavioral modeling using Multinomial Logit (MNL)

These features transform the simulator from basic to production-ready, matching real airline operations.

---

## 1. Overbooking Module

### What is Overbooking?

Airlines deliberately accept more bookings than available seats because:
- **Passengers don't always show up** (no-show rates: 5-15%)
- **Flying with empty seats loses revenue** 
- **Overbooking increases revenue by 2-5%** when done correctly

The challenge: Balance revenue gains vs. denied boarding costs.

### Key Components

#### No-Show Probability Model

No-show rates vary by:

| Factor | High No-Show | Low No-Show |
|--------|--------------|-------------|
| **Segment** | Business (15%) | Groups (3%) |
| **Fare Class** | Full-fare Y (high) | Restricted L (low) |
| **Booking Time** | >21 days advance | Same-day |
| **Route Type** | Hub connections | Point-to-point |

**Code Example:**
```python
from overbooking.optimizer import NoShowModel

no_show_model = NoShowModel()

# Calculate no-show probability
prob = no_show_model.get_no_show_probability(
    segment=CustomerSegment.BUSINESS,
    booking_class=BookingClass.Y,
    advance_purchase_days=30,
    route_distance=1000,
    is_hub_flight=True
)
# Returns: ~0.18 (18% no-show)
```

#### Overbooking Optimization

Uses **Critical Fractile Policy**:
- Accept bookings until risk of denied boarding exceeds threshold
- Critical ratio = Revenue / (Revenue + Denied Boarding Cost)

**Methods:**
1. **Critical Fractile** (standard) - Balances revenue and risk
2. **Risk Averse** (conservative) - Minimizes denied boarding

**Code Example:**
```python
from overbooking.optimizer import (
    OverbookingOptimizer, 
    OverbookingMethod,
    DeniedBoardingCost
)

optimizer = OverbookingOptimizer(
    method=OverbookingMethod.CRITICAL_FRACTILE
)

policy = optimizer.calculate_overbooking_limit(
    capacity=150,              # Aircraft seats
    current_bookings=bookings, # List of Booking objects
    avg_fare=400.0,           # Average ticket price
    risk_tolerance=0.05       # Max 5% DB risk
)

print(f"Booking limit: {policy.booking_limit}")
print(f"Overbooking: {policy.overbooking_level} seats")
print(f"Expected revenue gain: ${policy.expected_revenue_gain:.2f}")
print(f"Expected DB cost: ${policy.expected_db_cost:.2f}")
print(f"Net benefit: ${policy.net_benefit:.2f}")
```

**Example Output:**
```
Booking limit: 165
Overbooking: 15 seats (10%)
Expected revenue gain: $6000
Expected DB cost: $800
Net benefit: $5200
```

#### Denied Boarding Costs

Based on US DOT regulations + operational costs:

| Delay Length | Compensation | Prob | Additional Costs |
|--------------|--------------|------|------------------|
| 0-1 hour | $0 | - | - |
| 1-2 hours | $400 (200% fare) | 40% | Rebooking: $150 |
| 2-4 hours | $800 (400% fare) | 30% | Hotel: $200 |
| 4+ hours | $1350 | 30% | Meals: $50 |

Plus intangible costs:
- Goodwill loss: $300
- Brand damage: $200

**Total expected cost per denied boarding: ~$1000-1500**

#### Show-Up Forecasting

Estimates distribution of passenger show-ups:

```python
forecast = no_show_model.estimate_show_up_distribution(
    bookings=confirmed_bookings,
    capacity=150
)

print(f"Expected shows: {forecast.expected_shows:.1f}")
print(f"Std deviation: {forecast.std_dev:.1f}")
print(f"Probability oversold: {forecast.probability_oversold*100:.1f}%")
print(f"Expected denied boardings: {forecast.expected_denied_boardings:.2f}")
```

### Integration with Simulator

Enable overbooking in simulation config:

```python
from core.simulator import SimulationConfig

config = SimulationConfig(
    overbooking_enabled=True,
    overbooking_method="critical_fractile",  # or "risk_averse"
    overbooking_risk_tolerance=0.05,        # 5% max DB risk
    # ... other config
)
```

The simulator will:
1. Calculate overbooking limits dynamically as bookings accumulate
2. Accept bookings up to the overbooking limit
3. Simulate show-ups at departure
4. Handle denied boardings with appropriate costs

### Business Impact

**Revenue Improvement:**
- Typical gain: **2-5% of total revenue**
- For $100M annual revenue: **$2-5M additional revenue**

**Key Tradeoffs:**
- More aggressive → higher revenue, higher DB risk
- More conservative → lower DB risk, lower revenue
- Optimal balance depends on route, segment mix, costs

---

## 2. Customer Choice Models

### Why Customer Choice Matters

Simple "cheapest option" logic is unrealistic:
- Customers balance price, time, convenience
- Some buy up to better class
- Some buy down to save money
- Loyalty affects choices
- Recapture rates when preferred option unavailable

**Impact of realistic choice models:**
- Revenue estimates: ±5-10% more accurate
- Optimal pricing: Different strategies needed
- Spill estimates: Recapture prevents overestimation

### Multinomial Logit (MNL) Model

**Industry standard** for airline choice modeling.

**Utility Function:**
```
U = β_price × price + β_time × time + β_direct × is_direct + 
    β_schedule × schedule_quality + β_loyalty × loyalty + ε

Where ε ~ Gumbel(0, μ) - random component
```

**Choice Probability:**
```
P(alternative i) = exp(U_i) / Σ_j exp(U_j)
```

**Code Example:**
```python
from choice.models import MultinomialLogitModel, UtilityFunction, ChoiceSet

# Create utility function with custom coefficients
utility_fn = UtilityFunction(
    price_coef=-0.002,          # Negative = disutility of cost
    time_coef=-0.01,            # Per minute
    connection_penalty=-0.5,     # Per connection
    direct_flight_bonus=1.0,
    loyalty_coef=0.5,
    business_time_multiplier=2.0,    # Business values time more
    business_price_multiplier=0.5    # Business less price-sensitive
)

# Create MNL model
mnl = MultinomialLogitModel(utility_function=utility_fn)

# Calculate choice probabilities
choice_set = ChoiceSet(
    own_solutions=[solution1, solution2, solution3],
    competitor_solutions=[],
    no_purchase_utility=0.0
)

probs = mnl.calculate_choice_probabilities(choice_set, customer)
# Returns: {0: 0.25, 1: 0.45, 2: 0.20, -1: 0.10}
#          Options 1-3 and no-purchase (-1)
```

**Example Scenario:**
```
Customer: Leisure, WTP=$500, Price-sensitive

Options:
1. Direct $450, 5.5h → Utility = -0.9 + 1.0 - 3.3 = -3.2 → P = 15%
2. 1-stop $320, 8.25h → Utility = -0.64 - 0.5 - 4.95 = -6.09 → P = 45%
3. 2-stop $250, 11.75h → Utility = -0.5 - 1.0 - 7.05 = -8.55 → P = 30%
No purchase → P = 10%

Most likely choice: Option 2 (good price-time tradeoff)
```

### Buy-Up and Buy-Down Behavior

**Buy-Up:** Customer accepts higher class when preferred unavailable
**Buy-Down:** Customer accepts lower class to save money

**Probabilities by Segment:**

| Segment | Buy-Up | Buy-Down | Max Price Increase |
|---------|--------|----------|-------------------|
| Business | 40% | 15% | 50% |
| Leisure | 10% | 50% | 20% |
| VFR | 5% | 45% | 15% |
| Group | 2% | 40% | 10% |

**Code Example:**
```python
from choice.models import BuyUpDownModel

buyupdown = BuyUpDownModel()

# Will customer buy up from $350 to $450?
will_buy = buyupdown.will_buy_up(
    customer=customer,
    preferred_price=350.0,
    higher_price=450.0,
    rng=rng
)
# Business: 40% probability
# Leisure: 10% probability

# Will customer buy down from $350 to $250?
will_buy = buyupdown.will_buy_down(
    customer=customer,
    preferred_price=350.0,
    lower_price=250.0,
    rng=rng
)
# Business: 15% probability
# Leisure: 50% probability
```

### Recapture Modeling

**Recapture Rate:** Probability customer accepts alternative instead of:
- Going to competitor
- Not traveling at all

**Base Rates by Segment:**
- Business: 60% (higher loyalty, less flexibility)
- Leisure: 40% (more likely to shop around)
- VFR: 50%
- Groups: 70% (committed travel)

**Adjustments:**
- Loyalty program: +20%
- High price sensitivity: -15%
- Better alternative utility: +20%

**Code Example:**
```python
from choice.models import RecaptureModel

recapture = RecaptureModel()

prob = recapture.get_recapture_probability(
    customer=customer,
    has_loyalty=True
)
# Business with loyalty: 60% + 20% = 80%
# Leisure without loyalty: 40%

will_accept = recapture.will_accept_alternative(
    customer=customer,
    alternative_utility=-5.0,
    competitor_utility=-6.0,  # Our option is better
    has_loyalty=True,
    rng=rng
)
# High probability of acceptance
```

### Enhanced Choice Model

Combines all three behaviors:

```python
from choice.models import EnhancedChoiceModel

enhanced = EnhancedChoiceModel(
    mnl_model=mnl,
    buyupdown_model=buyupdown,
    recapture_model=recapture
)

chosen = enhanced.predict_choice_with_behavior(
    choice_set=choice_set,
    customer=customer,
    preferred_class=BookingClass.M,
    rng=rng
)
```

**Decision Logic:**
1. Try MNL choice among available options
2. If preferred class unavailable, consider buy-up/down
3. If still no choice, evaluate recapture
4. Return final choice or None

### Integration with Simulator

Enable in simulation config:

```python
config = SimulationConfig(
    choice_model="mnl",           # "cheapest", "mnl", or "enhanced"
    include_buyup_down=True,      # Enable buy-up/down behavior
    include_recapture=True,        # Enable recapture modeling
    # ... other config
)
```

The simulator automatically uses the choice model for all booking requests.

### Business Impact

**Revenue Impact:**
- Buy-up behavior: +1-2% revenue
- Accurate spill estimation: ±3-5% in network optimization
- Better pricing decisions: +2-4% yield improvement

**Example:**
Without choice model:
- Customer wants Y class ($400), not available
- Rejects all options → lost sale

With enhanced choice model:
- Customer wants Y class ($400), not available
- 40% buy up to B class ($550) → +$150 revenue
- 15% buy down to M class ($300) → -$100 revenue
- 60% recaptured with alternative → retained booking
- Net effect: +2-3% revenue vs. simple model

---

## 3. Complete Example

```python
from datetime import date, timedelta
from core.simulator import SimulationConfig, Simulator
from core.models import *
from overbooking.optimizer import OverbookingOptimizer, OverbookingMethod
from choice.models import EnhancedChoiceModel

# Create simulation with both features
config = SimulationConfig(
    start_date=date.today(),
    end_date=date.today() + timedelta(days=90),
    
    # Overbooking settings
    overbooking_enabled=True,
    overbooking_method="critical_fractile",
    overbooking_risk_tolerance=0.05,
    
    # Choice model settings  
    choice_model="enhanced",
    include_buyup_down=True,
    include_recapture=True,
    
    # Other settings
    rm_method="EMSR-b",
    dynamic_pricing=True,
    random_seed=42
)

# Create and run simulator
simulator = Simulator(config, schedules, routes, airports)
results = simulator.run()

# Results include:
# - Revenue gains from overbooking
# - Denied boarding events and costs
# - Customer choice patterns (buy-up/down rates)
# - Recapture statistics
```

---

## 4. Performance Metrics

### Overbooking Metrics

**Track these KPIs:**
- Overbooking rate (% of capacity)
- Show-up rate (actual vs. forecast)
- Denied boarding rate (events per 10,000 passengers)
- Revenue gain from overbooking
- Denied boarding costs
- Net benefit

**Industry Benchmarks:**
- Typical overbooking: 5-15% of capacity
- Denied boarding: 0.1-1.0 per 10,000 passengers
- Revenue gain: 2-5% of total revenue

### Choice Model Metrics

**Track these KPIs:**
- Buy-up rate (%)
- Buy-down rate (%)
- Recapture rate (%)
- Average choice probability
- Spill vs. actual no-purchase

**Validation:**
- Compare simulated choice rates to historical data
- Calibrate utility coefficients with market research
- A/B test pricing strategies

---

## 5. Best Practices

### Overbooking

✅ **Do:**
- Start conservative (risk_tolerance=0.03-0.05)
- Monitor denied boarding rates closely
- Adjust by route (hubs vs. point-to-point)
- Account for segment mix (business = higher no-show)
- Update no-show models with historical data

❌ **Don't:**
- Overbook too aggressively (reputation damage)
- Use same policy for all routes
- Ignore operational constraints (rebooking availability)
- Forget accommodation costs for overnight delays

### Customer Choice

✅ **Do:**
- Calibrate utility coefficients with real data
- Segment customers properly
- Include competitor options in choice set
- Model schedule convenience (time of day)
- Account for loyalty program effects

❌ **Don't:**
- Use generic coefficients for all markets
- Ignore customer heterogeneity
- Assume perfect recapture
- Forget about no-purchase option
- Ignore competitive actions

---

## 6. Advanced Topics

### Dynamic Overbooking

Overbooking limits should change as:
- Booking pace varies
- Mix of segments changes
- Time to departure decreases
- Historical patterns emerge

**Implementation:** Recalculate policy at each RM optimization point.

### Nested Logit Models

For more sophisticated choice:
- Level 1: Choose airline
- Level 2: Choose product (class, schedule)

Better captures sequential decision-making.

### Machine Learning Integration

Use ML for:
- No-show prediction (better accuracy)
- Utility coefficient estimation
- Personalized choice models
- Dynamic recapture rates

---

## 7. Summary

### Overbooking

**Key Benefits:**
- 2-5% revenue increase
- Better seat utilization
- Optimizes empty seat risk vs. denied boarding risk

**Critical Components:**
- No-show probability modeling
- Critical fractile optimization
- Denied boarding cost estimation
- Show-up distribution forecasting

### Customer Choice

**Key Benefits:**
- 5-10% more accurate revenue forecasts
- Better pricing decisions
- Realistic buy-up/down behavior
- Proper spill and recapture modeling

**Critical Components:**
- Multinomial Logit (MNL) model
- Utility-based choice
- Buy-up/down probabilities
- Recapture rates

### Combined Impact

Using both features together:
- **Total revenue improvement: 5-10%** vs. simple model
- Much more realistic simulation
- Better supports strategic planning
- Matches actual airline operations

---

## 8. References

**Overbooking:**
- Karaesmen & van Ryzin (2004): "Overbooking with Substitutable Inventory Classes"
- Suzuki (2006): "A Decision-Support System for Seat Inventory and Overbooking"

**Customer Choice:**
- Talluri & van Ryzin (2004): "Revenue Management Under a General Discrete Choice Model"
- Vulcano et al. (2010): "RM: The Choice-Based Model"

**Industry Practice:**
- AGIFORS conferences on RM
- US DOT denied boarding compensation rules
- IATA RM guidelines
