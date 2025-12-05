# Full Year Seasonality & DTD Analysis

## Simulation Overview
- **Period:** Jan 1, 2025 - Dec 31, 2025 (365 days)
- **Route:** JFK-LAX (AA100)
- **Total Bookings:** 22,200
- **Total Revenue:** $10.6M
- **Load Factor:** 28.2%

## Seasonality Findings

### 1. New Year's (Jan 1)
- **Behavior:** High percentage of close-in bookings (0-7 days).
- **Stats:** Jan 1 had 20% close-in bookings.
- **Interpretation:** Last-minute travel plans or return trips.

### 2. Valentine's Day (Feb 14)
- **Behavior:** Strong advance booking behavior.
- **Stats:** <5% close-in bookings. >90% booked 30+ days out.
- **Interpretation:** Planned leisure travel.

### 3. Easter (Apr 20)
- **Behavior:** Similar to Valentine's, high advance bookings.
- **Stats:** Apr 20 had 60% bookings in the 31-60 day window.

### 4. Independence Day (Jul 4)
- **Behavior:** High volume surrounding the holiday.
- **Stats:** Jul 3 had 106 bookings (vs ~50 average). Jul 4 itself had 71.
- **Interpretation:** Peak summer travel.

### 5. Thanksgiving (Nov 27)
- **Behavior:** Classic "trough" on the holiday, "peak" around it.
- **Stats:** 
    - Nov 27 (Day of): 32 bookings (Low)
    - Nov 26 (Day before): 76 bookings (High)
    - Nov 28 (Day after): 77 bookings (High)
- **Interpretation:** People fly to be there *for* the holiday, not *on* it.

### 6. Christmas (Dec 25)
- **Behavior:** Extreme trough/peak pattern.
- **Stats:**
    - Dec 25: 22 bookings (Lowest in period)
    - Dec 26: 122 bookings (Highest in period)
- **Interpretation:** No one flies on Christmas Day; everyone flies the day after.

## Conclusion
The simulation successfully captures complex seasonality patterns:
1.  **Volume Seasonality:** Peaks around holidays, troughs on the holiday itself (for family holidays).
2.  **Booking Curve Seasonality:** Distinct "Close-in" vs "Advance" profiles depending on the holiday type.

## Visualizations
- `simulation_results/dtd_analysis_stacked.png`: Stacked bar chart showing daily booking composition with holiday markers.
- `simulation_results/dtd_heatmap.png`: Heatmap showing booking intensity by DTD window and date.
