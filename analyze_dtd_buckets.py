
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

def analyze_dtd_buckets():
    print("Analyzing DTD buckets...")
    
    file_path = "/Users/ajit/Desktop/Github Projects/pyairline_rm/simulation_results/bookings-5.csv"
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    df = pd.read_csv(file_path)
    
    # Filter for AA100 if multiple flights exist (though our sim was single flight mode)
    df = df[df['flight_code'] == 'AA100'].copy()
    
    # Convert dates
    df['booking_time'] = pd.to_datetime(df['booking_time'])
    df['departure_date'] = pd.to_datetime(df['departure_date'])
    
    # Calculate DTD (Days To Departure)
    # DTD = (Departure Date - Booking Date).days
    # Note: booking_time has time, departure_date is just date (midnight).
    # We should normalize booking_time to date for DTD calc or just take the delta days.
    df['booking_date'] = df['booking_time'].dt.normalize()
    df['dtd'] = (df['departure_date'] - df['booking_date']).dt.days
    
    # Define Buckets
    def get_dtd_bucket(dtd):
        if dtd <= 7:
            return "0-7 Days (Close-in)"
        elif dtd <= 14:
            return "8-14 Days"
        elif dtd <= 30:
            return "15-30 Days"
        elif dtd <= 60:
            return "31-60 Days"
        else:
            return "60+ Days (Advance)"

    df['dtd_bucket'] = df['dtd'].apply(get_dtd_bucket)
    
    # Order for plotting
    bucket_order = [
        "0-7 Days (Close-in)",
        "8-14 Days",
        "15-30 Days",
        "31-60 Days",
        "60+ Days (Advance)"
    ]
    
    # Group by Departure Date and Bucket
    daily_dtd = df.groupby(['departure_date', 'dtd_bucket']).size().unstack(fill_value=0)
    
    # Reorder columns
    daily_dtd = daily_dtd.reindex(columns=bucket_order, fill_value=0)
    
    # Plotting
    plt.figure(figsize=(15, 8))
    
    # Stacked Bar Chart
    daily_dtd.plot(kind='bar', stacked=True, figsize=(15, 8), width=1.0, colormap='viridis')
    
    plt.title('Daily Bookings by Days-To-Departure (DTD) Window')
    plt.xlabel('Departure Date')
    plt.ylabel('Number of Bookings')
    plt.legend(title='Booking Window')

    # Add markers for holidays
    holidays = {
        'New Year': '2025-01-01',
        'Valentine': '2025-02-14',
        'Easter': '2025-04-20',
        'Memorial Day': '2025-05-26',
        'Independence': '2025-07-04',
        'Labor Day': '2025-09-01',
        'Thanksgiving': '2025-11-27',
        'Christmas': '2025-12-25'
    }
    
    # Get all dates from the index to map dates to x-axis positions
    all_dates = daily_dtd.index
    y_max = daily_dtd.sum(axis=1).max()
    
    for name, date_str in holidays.items():
        h_date = pd.to_datetime(date_str)
        if h_date in all_dates:
            # Find the integer location (index) of the date
            idx = all_dates.get_loc(h_date)
            
            # Add a vertical line
            plt.axvline(x=idx, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            
            # Add a text label
            plt.text(idx, y_max + 5, name, rotation=90, color='red', ha='center', va='bottom', fontweight='bold')
    
    # Format X-axis to show fewer labels
    ax = plt.gca()
    
    # Since bar plot uses categorical x-axis, we need to fix the labels
    # Get current ticks and labels
    ticks = ax.get_xticks()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    
    # Convert labels back to datetime to format them nicely
    # The labels from groupby are timestamps
    formatted_labels = []
    for label in labels:
        try:
            dt = pd.to_datetime(label)
            formatted_labels.append(dt.strftime('%b %d'))
        except:
            formatted_labels.append(label)
            
    # Show only every 7th label to avoid clutter
    n = 7
    for i, label in enumerate(ax.get_xticklabels()):
        if i % n != 0:
            label.set_visible(False)
        else:
            label.set_text(formatted_labels[i])
            
    plt.tight_layout()
    plt.savefig('simulation_results/dtd_analysis_stacked.png')
    print("Plot saved to simulation_results/dtd_analysis_stacked.png")
    
    # --- Heatmap Analysis ---
    # Pivot for heatmap: Rows=Departure Date, Cols=DTD Bucket
    plt.figure(figsize=(12, 10))
    sns.heatmap(daily_dtd, cmap="YlGnBu", cbar_kws={'label': 'Bookings'})
    plt.title('Heatmap of Bookings by Departure Date and DTD Window')

    # Add markers to heatmap
    for name, date_str in holidays.items():
        h_date = pd.to_datetime(date_str)
        if h_date in all_dates:
            idx = all_dates.get_loc(h_date)
            # Add horizontal line
            plt.axhline(y=idx, color='red', linestyle='--', alpha=0.8, linewidth=1.5)
            # Add text label on the right side
            plt.text(len(bucket_order) + 0.2, idx, name, color='red', va='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig('simulation_results/dtd_heatmap.png')
    print("Heatmap saved to simulation_results/dtd_heatmap.png")

    # --- Holiday Analysis with DTD ---
    print("\nHoliday DTD Composition (Window +/- 2 days):")
    holidays = {
        'New Year': '2025-01-01',
        'Valentine': '2025-02-14',
        'Easter': '2025-04-20',
        'Memorial Day': '2025-05-26',
        'Independence': '2025-07-04',
        'Labor Day': '2025-09-01',
        'Thanksgiving': '2025-11-27',
        'Christmas': '2025-12-25'
    }
    
    for name, date_str in holidays.items():
        h_date = pd.to_datetime(date_str)
        start_window = h_date - pd.Timedelta(days=2)
        end_window = h_date + pd.Timedelta(days=2)
        
        print(f"\n--- {name} Window ({start_window.date()} to {end_window.date()}) ---")
        
        window_data = daily_dtd.loc[start_window:end_window]
        
        for dt in window_data.index:
            row = window_data.loc[dt]
            total = row.sum()
            print(f"Date: {dt.date()} | Total: {total}")
            for bucket in bucket_order:
                count = row[bucket]
                pct = (count / total * 100) if total > 0 else 0
                if count > 0:
                    print(f"    {bucket}: {count} ({pct:.1f}%)")

if __name__ == "__main__":
    analyze_dtd_buckets()
