# DTD Window Analysis

## Overview
This analysis breaks down the booking composition for flight `AA100` (JFK-LAX) by "Days To Departure" (DTD) windows. This helps understand *when* customers are booking relative to the flight date.

## Methodology
- **Data Source:** Downloaded bookings from the 6-month API simulation.
- **Buckets:**
  - **0-7 Days:** Close-in bookings (often Business/Distressed)
  - **8-14 Days:** Near-term
  - **15-30 Days:** Medium-term
  - **31-60 Days:** Standard advance purchase
  - **60+ Days:** Long-term advance

## Key Findings

### 1. General Trend
- The majority of bookings for this route appear to happen in the **31-60 day** and **60+ day** windows.
- This suggests a strong leisure component or a simulation configuration where business demand (close-in) is priced out or lower volume.

### 2. Holiday Composition
We analyzed the booking windows for specific holidays:

| Holiday | Date | Total Bookings | Close-in (0-7d) | Advance (31d+) | Insight |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **New Year's** | Jan 1 | 55 | 18.2% | 54.6% | Higher close-in % than other holidays, possibly last-minute travel. |
| **Valentine's** | Feb 14 | 59 | 1.7% | 93.2% | Almost entirely booked in advance. |
| **Easter** | Apr 20 | 61 | 3.3% | 95.1% | Heavily advance booked. |
| **Memorial Day** | May 26 | 37 | 5.4% | 89.2% | Mostly advance, low overall volume. |

### 3. Visualizations
- **Stacked Bar Chart:** `simulation_results/dtd_analysis_stacked.png` - Shows the daily volume colored by booking window.
- **Heatmap:** `simulation_results/dtd_heatmap.png` - Provides a density view of booking windows over time.

## Conclusion
The simulation successfully models different booking curves. The high proportion of advance bookings for holidays aligns with typical leisure travel behavior, where passengers book well ahead of time to secure seats for peak periods.
