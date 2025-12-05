# Seasonality and Holiday Simulation Analysis

## Overview
A 6-month simulation (Jan 1, 2025 - Jun 30, 2025) was conducted to verify the implementation of seasonality and dynamic holiday logic.

## Methodology
- **Route:** JFK-LAX (Daily B777 flight)
- **Base Demand:** 150 passengers/day (mean)
- **Simulation Period:** 181 days
- **Seasonality Factors:**
  - Monthly multipliers (low in Jan/Feb, high in Summer)
  - Day-of-week multipliers (high Fri/Sun, low Tue/Wed)
  - Holiday multipliers (specific to each holiday)

## Holiday Analysis Results

The simulation confirmed correct demand modulation for fixed and dynamic holidays:

### 1. New Year's (Fixed)
- **Jan 1:** 71 requests (Low demand, ~0.6x multiplier)
- **Jan 2-3:** Surge in return travel (142-223 requests)

### 2. Valentine's Day (Fixed)
- **Feb 14:** 162 requests (Minor bump, ~1.1x multiplier)

### 3. Easter (Dynamic - April 20, 2025)
- **Good Friday (Apr 18):** 233 requests (Peak travel, 1.3x multiplier)
- **Easter Sunday (Apr 20):** 133 requests (Low travel, 0.8x multiplier)
- **Easter Monday (Apr 21):** 173 requests (Return travel, 1.2x multiplier)

### 4. Memorial Day (Dynamic - May 26, 2025)
- **Friday Before (May 23):** 237 requests (Peak summer kickoff, 1.3x multiplier)
- **Memorial Day (May 26):** 117 requests (Low travel on holiday, 0.7x multiplier)

## Conclusion
The demand generation logic correctly handles:
- **Dynamic Dates:** Easter (Computus algorithm) and Memorial Day (relative dates) were correctly identified and simulated.
- **Demand Shaping:** Multipliers effectively created peaks and troughs around holidays.
- **Interaction:** Holiday factors successfully layered on top of weekly and monthly seasonality.

## Artifacts
- `run_long_simulation.py`: Script to reproduce the 6-month simulation.
- `analyze_long_simulation.py`: Script to generate demand plots and statistics.
- `simulation_results/seasonality_analysis.png`: Visualization of daily demand trends.
