
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import date, timedelta

def analyze_seasonality():
    print("Analyzing seasonality from simulation results...")
    
    # Load booking requests
    try:
        df = pd.read_csv('simulation_results/booking_requests.csv')
    except FileNotFoundError:
        print("Error: simulation_results/booking_requests.csv not found.")
        return

    # Convert departure_date to datetime
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    
    # Count requests per departure date
    daily_demand = df.groupby('departure_date').size().reset_index(name='requests')
    
    # Plot
    plt.figure(figsize=(15, 8))
    plt.plot(daily_demand['departure_date'], daily_demand['requests'], label='Daily Requests', alpha=0.7)
    
    # Calculate 7-day rolling average to show trend
    daily_demand['rolling_avg'] = daily_demand['requests'].rolling(window=7).mean()
    plt.plot(daily_demand['departure_date'], daily_demand['rolling_avg'], color='red', linewidth=2, label='7-Day Moving Avg')
    
    # Mark Holidays
    holidays = {
        'New Year': '2025-01-01',
        'Valentine': '2025-02-14',
        'Easter': '2025-04-20',
        'Memorial Day': '2025-05-26'
    }
    
    for name, date_str in holidays.items():
        holiday_date = pd.to_datetime(date_str)
        if holiday_date in daily_demand['departure_date'].values:
            demand_val = daily_demand.loc[daily_demand['departure_date'] == holiday_date, 'requests'].values[0]
            
            # Convert to matplotlib date for annotation
            holiday_num = mdates.date2num(holiday_date)
            
            plt.annotate(name, 
                         xy=(holiday_num, demand_val), 
                         xytext=(holiday_num, demand_val + 50),
                         arrowprops=dict(facecolor='black', shrink=0.05),
                         horizontalalignment='center')
            
            print(f"Holiday {name} ({date_str}): {demand_val} requests")

    plt.title('Daily Demand - Jan to Jun 2025')
    plt.xlabel('Date')
    plt.ylabel('Number of Requests')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.tight_layout()
    plt.savefig('simulation_results/seasonality_analysis.png')
    print("Plot saved to simulation_results/seasonality_analysis.png")
    
    # Print some stats around holidays
    print("\nDetailed Holiday Analysis:")
    for name, date_str in holidays.items():
        h_date = pd.to_datetime(date_str)
        start_window = h_date - timedelta(days=3)
        end_window = h_date + timedelta(days=3)
        
        window_data = daily_demand[(daily_demand['departure_date'] >= start_window) & 
                                   (daily_demand['departure_date'] <= end_window)]
        
        print(f"\n{name} Window ({start_window.date()} to {end_window.date()}):")
        print(window_data.to_string(index=False))

if __name__ == "__main__":
    analyze_seasonality()
