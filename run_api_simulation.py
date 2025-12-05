
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

API_URL = "http://localhost:8000"

def run_simulation():
    print("Starting simulation via API...")
    
    # 1. Start Simulation
    payload = {
        "start_date": "2025-01-01",
        "end_date": "2025-12-31", # Full year to see all holidays
        "rm_method": "EMSR-b",
        "enable_holidays": True,
        "selected_airlines": ["AA"], # Keep it simple for analysis
        "single_flight_mode": True,  # Focus on JFK-LAX
        "demand_multiplier": 1.0
    }
    
    try:
        response = requests.post(f"{API_URL}/simulations", json=payload)
        response.raise_for_status()
        data = response.json()
        sim_id = data["simulation_id"]
        print(f"Simulation started with ID: {sim_id}")
    except requests.exceptions.RequestException as e:
        print(f"Error starting simulation: {e}")
        if response.text:
            print(f"Response: {response.text}")
        return

    # 2. Poll Status
    while True:
        try:
            status_res = requests.get(f"{API_URL}/simulations/{sim_id}/status")
            status_res.raise_for_status()
            status_data = status_res.json()
            
            status = status_data["status"]
            progress = status_data["progress"]
            message = status_data["message"]
            
            print(f"Status: {status} ({progress}%) - {message}")
            
            if status == "completed":
                break
            if status == "failed":
                print("Simulation failed.")
                return
            
            time.sleep(2)
        except requests.exceptions.RequestException as e:
            print(f"Error polling status: {e}")
            return

    # 3. Get Results & Download Data
    print("Fetching results...")
    try:
        results_res = requests.get(f"{API_URL}/simulations/{sim_id}/results")
        results_res.raise_for_status()
        results = results_res.json()
        
        print(f"Total Revenue: ${results['total_revenue']:,.2f}")
        print(f"Total Bookings: {results['total_bookings']:,}")
        
        # Download bookings from DB export endpoint
        print("Downloading bookings from database...")
        bookings_url = f"/simulations/{sim_id}/db/bookings/csv"
        
        try:
            file_res = requests.get(f"{API_URL}{bookings_url}")
            file_res.raise_for_status()
            
            os.makedirs("simulation_results", exist_ok=True)
            output_path = "simulation_results/downloaded_bookings.csv"
            
            with open(output_path, "wb") as f:
                f.write(file_res.content)
            print(f"Saved bookings to {output_path}")
            
            analyze_data(output_path)
        except requests.exceptions.RequestException as e:
            print(f"Error downloading bookings: {e}")
            # Fallback to checking exported_files just in case
            exported_files = results.get("exported_files", {})
            print("Available exported files:", exported_files.keys())

    except requests.exceptions.RequestException as e:
        print(f"Error fetching results: {e}")

def analyze_data(file_path):
    print("\nAnalyzing downloaded data...")
    try:
        df = pd.read_csv(file_path)
        
        # Convert dates
        df['booking_time'] = pd.to_datetime(df['booking_time'])
        df['departure_date'] = pd.to_datetime(df['departure_date'])
        
        # 1. Daily Demand (Bookings by Departure Date)
        daily_bookings = df.groupby('departure_date').size().reset_index(name='bookings')
        
        plt.figure(figsize=(15, 8))
        plt.plot(daily_bookings['departure_date'], daily_bookings['bookings'], label='Daily Bookings', alpha=0.7)
        
        # Rolling average
        daily_bookings['rolling_avg'] = daily_bookings['bookings'].rolling(window=7).mean()
        plt.plot(daily_bookings['departure_date'], daily_bookings['rolling_avg'], color='red', linewidth=2, label='7-Day Avg')
        
        # Mark Holidays
        holidays = {
            'New Year': '2025-01-01',
            'Valentine': '2025-02-14',
            'Easter': '2025-04-20',
            'Memorial Day': '2025-05-26'
        }
        
        for name, date_str in holidays.items():
            holiday_date = pd.to_datetime(date_str)
            if holiday_date in daily_bookings['departure_date'].values:
                val = daily_bookings.loc[daily_bookings['departure_date'] == holiday_date, 'bookings'].values[0]
                
                # Convert to matplotlib date for annotation
                holiday_num = mdates.date2num(holiday_date)
                
                plt.annotate(name, 
                             xy=(holiday_num, val), 
                             xytext=(holiday_num, val + 20),
                             arrowprops=dict(facecolor='black', shrink=0.05),
                             horizontalalignment='center')
                print(f"Holiday {name} ({date_str}): {val} bookings")

        plt.title('Daily Bookings (Downloaded from API)')
        plt.xlabel('Departure Date')
        plt.ylabel('Number of Bookings')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        
        plt.tight_layout()
        plt.savefig('simulation_results/api_analysis.png')
        print("Analysis plot saved to simulation_results/api_analysis.png")
        
    except Exception as e:
        print(f"Error analyzing data: {e}")

if __name__ == "__main__":
    run_simulation()
