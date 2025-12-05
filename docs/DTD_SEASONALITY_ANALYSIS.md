# DTD Seasonality Analysis

## Overview
This analysis examines the booking behavior around major holidays for flight `AA100` (JFK-LAX). By analyzing the "Days To Departure" (DTD) buckets for the days surrounding each holiday, we can pinpoint seasonality effects not just in volume, but in booking behavior.

## Key Findings

### 1. New Year's Surge (Jan 1 - Jan 3)
- **Jan 1 (Holiday):** 55 bookings. High close-in bookings (18.2%), suggesting last-minute travel.
- **Jan 2 (Post-Holiday):** 95 bookings. Volume nearly doubles.
- **Jan 3 (Return Peak):** 135 bookings. Significant surge.
  - **Behavior:** The surge is driven by a mix of advance and close-in bookings, but the absolute number of close-in bookings increases (16 vs 10 on Jan 1), indicating strong demand across all segments.

### 2. Valentine's Day (Feb 14)
- **Feb 14 (Friday):** 59 bookings.
- **Feb 16 (Sunday):** 72 bookings.
- **Behavior:** Very low close-in bookings (<2%). This is a highly planned leisure holiday. Over 90% of bookings occur 30+ days in advance.

### 3. Easter (Apr 18 - Apr 21)
- **Apr 18 (Good Friday):** 93 bookings. Peak travel day.
- **Apr 20 (Easter Sunday):** 61 bookings. Dip.
- **Apr 21 (Easter Monday):** 65 bookings. Moderate return.
- **Behavior:** Similar to Valentine's, this is heavily advance-booked (>90% over 30 days out). The peak on Good Friday confirms the holiday travel pattern.

### 4. Memorial Day (May 26)
- **May 26 (Monday):** 37 bookings. Significant dip.
- **May 25 (Sunday):** 65 bookings.
- **May 27 (Tuesday):** 65 bookings.
- **Behavior:** The holiday itself is a low-travel day. The surrounding days show higher volume, consistent with people traveling before/after the long weekend.

## Conclusion
The simulation correctly captures seasonality:
1.  **Volume Peaks:** Clear surges are visible around holidays (e.g., Jan 3, Apr 18).
2.  **Booking Behavior:** Leisure-heavy holidays (Valentine's, Easter) show distinctively high advance purchase rates compared to the post-New Year period, which has a "back-to-work" mixed profile.
