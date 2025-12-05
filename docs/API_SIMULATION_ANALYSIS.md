# API Simulation & Analysis Report

## Overview
This report documents the execution of a simulation via the REST API, the retrieval of booking data, and the analysis of seasonality and holiday effects.

## Execution Details
- **Simulation ID:** `a7939fb5-1990-4b7b-9186-2d25ac17e14b`
- **Period:** Jan 1, 2025 - Jun 30, 2025
- **Route:** JFK-LAX (Single Flight Mode)
- **Airline:** American Airlines (AA)
- **Total Revenue:** ~$4.99 Million
- **Total Bookings:** 10,388

## Data Retrieval
Booking data was downloaded directly from the simulation database via the API endpoint:
`GET /simulations/{sim_id}/db/bookings/csv`

## Seasonality Analysis
The analysis of the downloaded booking data confirms that holiday demand logic is active and effective.

### Observed Holiday Bookings
| Holiday | Date | Bookings | Observation |
| :--- | :--- | :--- | :--- |
| **New Year's Day** | Jan 1 | 55 | Low demand (Holiday dip) |
| **Valentine's Day** | Feb 14 | 59 | Moderate demand |
| **Easter Sunday** | Apr 20 | 61 | Lower than surrounding peak days |
| **Memorial Day** | May 26 | 37 | Significant dip (Holiday effect) |

### Visualization
A plot of daily bookings has been generated at `simulation_results/api_analysis.png`, showing the daily fluctuation of bookings and highlighting the specific holiday dates.

## Conclusion
The API correctly orchestrates the simulation with holiday logic enabled. The data export functionality allows for external analysis of the simulation results, verifying that the core logic remains consistent when accessed via the web service layer.
