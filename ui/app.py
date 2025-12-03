import streamlit as st
import requests
import time
import pandas as pd
import plotly.express as px
from datetime import date, timedelta

# Configuration
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PyAirline RM Simulator",
    page_icon="✈️",
    layout="wide"
)

def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def main():
    st.title("✈️ PyAirline Revenue Management Simulator")
    
    # Sidebar - Configuration
    st.sidebar.header("Simulation Settings")
    
    # Check API connection
    if not check_api_health():
        st.error(f"⚠️ Cannot connect to API at {API_URL}. Please ensure the server is running.")
        st.stop()
    else:
        st.sidebar.success("✅ API Connected")

    with st.sidebar.form("sim_config"):
        st.subheader("Time Period")
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Start Date", date(2025, 12, 1))
        end_date = col2.date_input("End Date", date(2025, 12, 7))
        
        st.subheader("Market Parameters")
        demand_multiplier = st.slider("Demand Multiplier", 0.1, 5.0, 1.0, 0.1)
        demand_pattern = st.selectbox("Demand Pattern", ["default", "high_business", "high_leisure"])
        
        # Detailed Demand Inputs
        with st.expander("Detailed Demand Settings"):
            business_proportion = st.slider("Business Traveler %", 0.0, 1.0, 0.30, 0.05)
            business_wtp = st.number_input("Business WTP Mean ($)", value=800.0, step=50.0)
            leisure_wtp = st.number_input("Leisure WTP Mean ($)", value=300.0, step=50.0)
        
        st.subheader("Network & Airlines")
        selected_airlines = st.multiselect("Select Airlines", ["AA", "UA", "DL"], default=["AA", "UA", "DL"])
        single_flight_mode = st.checkbox("Single Flight Mode (JFK-LAX only)", False)
        
        st.subheader("RM Strategy")
        rm_method = st.selectbox("RM Method", ["EMSR-b", "EMSR-a", "Simple"])
        choice_model = st.selectbox("Choice Model", ["mnl", "enhanced", "cheapest", "custom"])
        
        # Custom Choice Parameters
        price_sensitivity = -0.002
        time_sensitivity = -0.01
        
        if choice_model == "custom":
            st.caption("Custom Choice Parameters")
            price_sensitivity = st.number_input("Price Sensitivity (Beta)", value=-0.002, step=0.001, format="%.4f")
            time_sensitivity = st.number_input("Time Sensitivity (Beta)", value=-0.01, step=0.01, format="%.3f")
        
        st.subheader("Features")
        dynamic_pricing = st.checkbox("Dynamic Pricing", True)
        overbooking = st.checkbox("Overbooking", True)
        
        st.subheader("Currency")
        customer_currency = st.selectbox("Customer Currency", ["USD", "EUR", "GBP", "JPY", "AUD", "CAD"])
        base_currency = st.selectbox("Base Currency", ["USD", "EUR"])
        
        submitted = st.form_submit_button("Run Simulation")

    # Main Content
    
    # Network Map
    st.subheader("Network Map")
    
    col_map1, col_map2 = st.columns(2)
    map_source = col_map1.text_input("Map Source (Code)", "JFK")
    map_destination = col_map2.text_input("Map Destination (Code)", "LAX")
    
    # Fetch airport data from API
    def get_airport_data(code):
        try:
            res = requests.get(f"{API_URL}/airports/{code}")
            if res.status_code == 200:
                return res.json()
        except:
            pass
        return None

    # Default airports for visualization - REMOVED
    # Only show user selected airports
    airport_list = []
    
    # Always try to fetch source and dest
    if map_source:
        src_data = get_airport_data(map_source)
        if src_data: airport_list.append(src_data)
    
    if map_destination:
        dst_data = get_airport_data(map_destination)
        if dst_data: airport_list.append(dst_data)
            
    if not airport_list:
        if map_source or map_destination:
             st.warning("Could not load airport data for the specified codes.")
        airports_df = pd.DataFrame(columns=['iata_code', 'latitude', 'longitude', 'city'])
    else:
        airports_df = pd.DataFrame(airport_list)

    # Create map
    # Always create a base map even if empty
    fig_map = px.scatter_geo(
        airports_df if not airports_df.empty else None,
        lat='latitude' if not airports_df.empty else None,
        lon='longitude' if not airports_df.empty else None,
        hover_name='city' if not airports_df.empty else None,
        text='iata_code' if not airports_df.empty else None,
        scope='world',
        title='Global Route Network',
        projection="natural earth",
        color_discrete_sequence=["#FF4B4B"]
    )
    
    # Customize map appearance
    fig_map.update_geos(
        showcountries=True, countrycolor="Black",
        showocean=True, oceancolor="LightBlue",
        showland=True, landcolor="LightGreen",
        showlakes=True, lakecolor="LightBlue",
        showrivers=True, rivercolor="LightBlue"
    )
    
    fig_map.update_layout(
        margin={"r":0,"t":30,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor= 'rgba(0,0,0,0)')
    )
    
    if not airports_df.empty:
        # Add lines for routes
        routes = []
        
        # Add user defined route
        if map_source and map_destination:
             # Check if we have coords
             has_src = any(a['iata_code'] == map_source for a in airport_list)
             has_dst = any(a['iata_code'] == map_destination for a in airport_list)
             
             if has_src and has_dst:
                 routes.append((map_source, map_destination))
             else:
                 st.warning(f"Could not find coordinates for {map_source} or {map_destination}")

        for origin, dest in routes:
            try:
                o_coords = airports_df[airports_df['iata_code'] == origin].iloc[0]
                d_coords = airports_df[airports_df['iata_code'] == dest].iloc[0]
                
                line_fig = px.line_geo(
                    lat=[o_coords['latitude'], d_coords['latitude']],
                    lon=[o_coords['longitude'], d_coords['longitude']],
                )
                line_trace = line_fig.data[0]
                line_trace.line = dict(color='red', width=2)
                fig_map.add_trace(line_trace)
            except IndexError:
                pass # Skip if airport not found in current df
    
    st.plotly_chart(fig_map, use_container_width=True)

    if submitted:
        if not selected_airlines:
            st.error("Please select at least one airline.")
            st.stop()
            
        # Prepare request
        payload = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "rm_method": rm_method,
            "choice_model": choice_model,
            "dynamic_pricing": dynamic_pricing,
            "overbooking": overbooking,
            "demand_multiplier": demand_multiplier,
            "random_seed": 42,
            "selected_airlines": selected_airlines,
            "demand_pattern": demand_pattern,
            "single_flight_mode": single_flight_mode,
            "customer_currency": customer_currency,
            "base_currency": base_currency,
            "business_proportion": business_proportion,
            "business_wtp": business_wtp,
            "leisure_wtp": leisure_wtp,
            "price_sensitivity": price_sensitivity,
            "time_sensitivity": time_sensitivity
        }
        
        try:
            # Start Simulation
            with st.spinner("Initializing simulation..."):
                response = requests.post(f"{API_URL}/simulations", json=payload)
                if response.status_code != 200:
                    st.error(f"Failed to start simulation: {response.text}")
                    st.stop()
                
                data = response.json()
                sim_id = data["simulation_id"]
            
            # Poll for progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                status_res = requests.get(f"{API_URL}/simulations/{sim_id}/status")
                if status_res.status_code != 200:
                    st.error("Lost connection to simulation.")
                    break
                
                status_data = status_res.json()
                progress = status_data["progress"]
                message = status_data["message"]
                state = status_data["status"]
                
                progress_bar.progress(progress)
                status_text.info(f"{message} ({progress}%)")
                
                if state == "completed":
                    st.success("Simulation Completed!")
                    display_results(sim_id)
                    break
                elif state == "failed":
                    st.error(f"Simulation Failed: {message}")
                    break
                
                time.sleep(1)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

def display_results(sim_id):
    res = requests.get(f"{API_URL}/simulations/{sim_id}/results")
    if res.status_code != 200:
        st.error("Could not retrieve results.")
        return
        
    results = res.json()
    
    # Top Level Metrics
    st.divider()
    st.header("Simulation Results")
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Revenue", f"${results['total_revenue']:,.2f}")
    m2.metric("Total Bookings", f"{results['total_bookings']:,}")
    m3.metric("Avg Load Factor", f"{results['avg_load_factor']:.1f}%")
    
    # Airline Comparison
    st.subheader("Airline Performance")
    
    metrics_df = pd.DataFrame(results['airline_metrics'])
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        fig_rev = px.bar(metrics_df, x='airline_name', y='revenue', 
                        title="Revenue by Airline", color='airline_name')
        st.plotly_chart(fig_rev, use_container_width=True)
        
    with c2:
        fig_lf = px.bar(metrics_df, x='airline_name', y='load_factor', 
                       title="Load Factor by Airline (%)", color='airline_name')
        st.plotly_chart(fig_lf, use_container_width=True)
    
    # Detailed Table
    st.dataframe(metrics_df)
    
    # Exported Files
    st.subheader("Downloads")
    if 'exported_files' in results and results['exported_files']:
        for desc, url in results['exported_files'].items():
            full_url = f"{API_URL}{url}"
            st.markdown(f"[{desc}]({full_url})")
    else:
        st.info("No files available for download.")

if __name__ == "__main__":
    main()
