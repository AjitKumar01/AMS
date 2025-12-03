# Simulation Data Export Documentation

## Overview

The PyAirline RM simulator automatically exports comprehensive data to CSV files for analysis. All exports are saved to the `simulation_results/` directory with detailed simulation logs.

## Configuration

Enable data export in your simulation config:

```python
config = SimulationConfig(
    export_csv=True,  # Enable CSV exports (default: True)
    export_log=True,  # Enable log file (default: True)
    output_dir="simulation_results"  # Output directory
)
```

## Generated Files

### 1. `bookings.csv` - Complete Booking Records

All accepted bookings with full details:

| Column | Description |
|--------|-------------|
| `booking_id` | Unique booking identifier |
| `booking_time` | When the booking was made |
| `departure_date` | Flight departure date |
| `flight_code` | Flight number (e.g., AA100) |
| `origin` | Origin airport code |
| `destination` | Destination airport code |
| `cabin_class` | F (First), J (Business), Y (Economy) |
| `booking_class` | Fare class (e.g., Y, B, M, K) |
| `customer_segment` | business, leisure, vfr, group |
| `party_size` | Number of passengers |
| `base_fare` | Base fare per passenger |
| `total_paid` | Total revenue from booking |
| `days_to_departure` | Days between booking and departure |
| `willingness_to_pay` | Customer's maximum WTP |
| `price_sensitivity` | Customer price sensitivity |
| `cancelled` | True if booking was cancelled |
| `cancellation_time` | When booking was cancelled (if applicable) |

**Use cases:**
- Revenue analysis by time period
- Customer behavior analysis
- Fare optimization studies
- Cancellation pattern analysis

### 2. `booking_requests.csv` - All Customer Requests

Every booking request (accepted and rejected):

| Column | Description |
|--------|-------------|
| `request_id` | Unique request identifier |
| `request_time` | When request was made |
| `departure_date` | Desired departure date |
| `origin` | Origin airport |
| `destination` | Destination airport |
| `preferred_cabin` | F, J, or Y |
| `customer_segment` | Customer type |
| `party_size` | Number of passengers |
| `willingness_to_pay` | Maximum WTP |
| `price_sensitivity` | Price sensitivity score |
| `time_sensitivity` | Time sensitivity score |
| `days_to_departure` | DTD at request time |
| `accepted` | True if booking was accepted |
| `rejection_reason` | Why rejected (if applicable) |

**Rejection reasons:**
- `no_availability` - Flight fully booked
- `price_too_high` - Fare exceeds WTP
- `customer_declined` - Customer rejected offer
- `no_suitable_flights` - No flights match criteria

**Use cases:**
- Demand analysis vs days-to-departure
- Spillage and recapture analysis
- Pricing optimization
- Rejection reason analysis

### 3. `dtd_analysis.csv` - Days-to-Departure Patterns

Aggregated booking patterns by weeks before departure:

| Column | Description |
|--------|-------------|
| `weeks_before_departure` | 0-12+ weeks |
| `days_range` | Day range (e.g., 0-6, 7-13) |
| `booking_count` | Number of bookings |
| `total_revenue` | Total revenue in bucket |
| `avg_fare` | Average fare per passenger |
| `business_pct` | % business travelers |
| `leisure_pct` | % leisure travelers |

**Use cases:**
- Booking curve analysis
- Demand forecasting
- Dynamic pricing strategies
- Customer segment mix by DTD

### 4. `segment_analysis.csv` - Customer Segment Performance

Performance metrics by customer segment:

| Column | Description |
|--------|-------------|
| `segment` | business, leisure, vfr, group |
| `booking_count` | Total bookings |
| `total_revenue` | Total revenue generated |
| `avg_revenue_per_booking` | Average booking value |
| `avg_willingness_to_pay` | Average WTP |
| `avg_party_size` | Average party size |
| `first_class_pct` | % First class bookings |
| `business_pct` | % Business class bookings |
| `economy_pct` | % Economy class bookings |

**Use cases:**
- Segment profitability analysis
- Targeted marketing insights
- Cabin mix optimization
- Revenue management by segment

### 5. `flight_metrics.csv` - Flight-Level Performance

Individual flight performance metrics:

| Column | Description |
|--------|-------------|
| `flight_id` | Unique flight identifier |
| `flight_code` | Flight number |
| `departure_date` | Date of departure |
| `origin` | Origin airport |
| `destination` | Destination airport |
| `capacity` | Total seats |
| `bookings` | Number of bookings |
| `load_factor` | Seats sold / capacity |
| `revenue` | Total flight revenue |
| `revenue_per_seat` | Revenue / capacity |
| `avg_fare` | Average fare per passenger |
| `cancellations` | Number of cancellations |

**Use cases:**
- Flight profitability analysis
- Load factor optimization
- Route performance comparison
- Capacity planning

### 6. `simulation_YYYYMMDD_HHMMSS.log` - Detailed Event Log

Timestamped log of all simulation events:
- Simulation initialization
- RM optimization runs
- Booking acceptances/rejections
- Snapshot events
- Final statistics

**Use cases:**
- Debugging simulation issues
- Understanding event sequencing
- Performance analysis
- Audit trail

## Analysis Examples

### 1. Booking Curve Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load DTD analysis
dtd = pd.read_csv('simulation_results/dtd_analysis.csv')

# Plot booking curve
plt.figure(figsize=(10, 6))
plt.bar(dtd['weeks_before_departure'], dtd['booking_count'])
plt.xlabel('Weeks Before Departure')
plt.ylabel('Number of Bookings')
plt.title('Booking Curve')
plt.show()
```

### 2. Revenue by Segment

```python
import pandas as pd

# Load segment analysis
segments = pd.read_csv('simulation_results/segment_analysis.csv')

# Calculate revenue contribution
segments['revenue_pct'] = (segments['total_revenue'] / 
                           segments['total_revenue'].sum() * 100)

print(segments[['segment', 'booking_count', 'total_revenue', 'revenue_pct']])
```

### 3. Acceptance Rate by DTD

```python
import pandas as pd

# Load requests
requests = pd.read_csv('simulation_results/booking_requests.csv')

# Group by DTD bucket
requests['dtd_bucket'] = pd.cut(requests['days_to_departure'], 
                                bins=[0, 7, 14, 21, 28, 100])

acceptance_by_dtd = requests.groupby('dtd_bucket')['accepted'].agg(['mean', 'count'])
print(acceptance_by_dtd)
```

### 4. Flight Load Factor Distribution

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load flight metrics
flights = pd.read_csv('simulation_results/flight_metrics.csv')

# Plot load factor distribution
plt.figure(figsize=(10, 6))
plt.hist(flights['load_factor'], bins=20, edgecolor='black')
plt.xlabel('Load Factor')
plt.ylabel('Number of Flights')
plt.title('Load Factor Distribution')
plt.axvline(flights['load_factor'].mean(), color='red', 
            linestyle='--', label=f"Mean: {flights['load_factor'].mean():.1f}%")
plt.legend()
plt.show()
```

## Data Quality Notes

1. **Timestamps**: All timestamps are in ISO 8601 format (`YYYY-MM-DDTHH:MM:SS`)
2. **Currency**: All revenue/fare values are in USD
3. **Percentages**: Load factors and acceptance rates are percentages (0-100)
4. **Missing Values**: Empty strings indicate missing/not applicable data
5. **Customer Segments**: Defined in `core/models.py` CustomerSegment enum

## Integration with BI Tools

The CSV exports are designed to work with:
- **Excel/Google Sheets**: Direct import for quick analysis
- **Tableau/Power BI**: Connect to CSV files for dashboards
- **Python/R**: Pandas/tidyverse for advanced analytics
- **SQL Databases**: Bulk load for historical analysis

## Performance Considerations

- CSV export happens at simulation end (minimal overhead during run)
- Large simulations (100k+ bookings) may take 1-2 seconds to export
- File sizes scale linearly with number of bookings
- Consider exporting to compressed formats for very large runs

## Custom Exports

To add custom CSV exports, extend the `DataExporter` class in `core/data_export.py`:

```python
def export_custom_analysis(self) -> str:
    """Export custom analysis."""
    filepath = self.output_dir / 'custom_analysis.csv'
    
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['column1', 'column2', 'column3'])
        
        # Your custom logic here
        for booking in self.bookings:
            # Process and write data
            pass
    
    return str(filepath)
```

Then call it in the simulator's `run()` method or use `export_all()`.
