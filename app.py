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
    
    if "sim_id" not in st.session_state:
        st.session_state.sim_id = None
    if "sim_completed" not in st.session_state:
        st.session_state.sim_completed = False
    
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
            st.markdown("### Customer Segments")
            business_proportion = st.slider("Business Traveler %", 0.0, 1.0, 0.30, 0.05)
            
            st.markdown("### Willingness To Pay (Mean)")
            col_wtp1, col_wtp2 = st.columns(2)
            business_wtp = col_wtp1.number_input("Business ($)", value=800.0, step=50.0)
            leisure_wtp = col_wtp2.number_input("Leisure ($)", value=300.0, step=50.0)
            
            st.markdown("### Advanced Parameters")
            forecast_smoothing = st.slider("Forecast Smoothing Factor (Alpha)", 0.0, 1.0, 0.2, 0.05)
            rm_frequency = st.selectbox("RM Optimization Frequency", ["daily", "12h", "6h", "realtime"])
        
        st.subheader("Network & Airlines")
        selected_airlines = st.multiselect("Select Airlines", ["AA", "UA", "DL"], default=["AA", "UA", "DL"])
        single_flight_mode = st.checkbox("Single Flight Mode (JFK-LAX only)", False)
        
        st.subheader("RM Strategy")
        rm_method = st.selectbox("RM Method", ["EMSR-b", "BidPrice", "EMSR-a", "Simple"])
        forecast_method = st.selectbox("Forecast Method", [
            "pickup", 
            "additive_pickup", 
            "multiplicative_pickup",
            "exponential_smoothing", 
            "historical_average",
            "neural_network",
            "xgboost",
            "ensemble"
        ])
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
        enable_holidays = st.checkbox("Enable Holidays", True, help="Include major holidays (New Year, Easter, Thanksgiving, etc.) in demand generation")
        personalization_enabled = st.checkbox("Enable Personalized Offers", False, help="Enable dynamic bundling and loyalty-based pricing")
        
        with st.expander("Advanced Configuration"):
            st.markdown("#### Optimization Settings")
            optimization_horizons_str = st.text_input("Optimization Horizons (days, comma-sep)", "30, 14, 7, 3, 1")
            price_update_freq = st.number_input("Price Update Frequency (hours)", value=6.0, min_value=1.0, step=1.0)
            
            st.markdown("#### Demand Generation")
            demand_gen_method = st.selectbox("Demand Generation Method", ["poisson", "stateful"])
            
            st.markdown("#### Overbooking Settings")
            overbooking_method = st.selectbox("Overbooking Method", ["critical_fractile", "risk_averse"])
            overbooking_risk = st.slider("Max Denied Boarding Risk", 0.0, 0.2, 0.05, 0.01)
            
            st.markdown("#### Customer Behavior")
            include_buyup_down = st.checkbox("Allow Buy-up/down", True)
            include_recapture = st.checkbox("Allow Recapture", True)
        
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
            
        # Map RM Frequency to hours
        rm_freq_hours = 24
        if rm_frequency == "12h":
            rm_freq_hours = 12
        elif rm_frequency == "6h":
            rm_freq_hours = 6
        elif rm_frequency == "realtime":
            rm_freq_hours = 1
            
        # Parse optimization horizons
        try:
            opt_horizons = [int(x.strip()) for x in optimization_horizons_str.split(",")]
        except:
            opt_horizons = [30, 14, 7, 3, 1]

        # Prepare request
        payload = {
            "start_date": str(start_date),
            "end_date": str(end_date),
            "rm_method": rm_method,
            "forecast_method": forecast_method,
            "choice_model": choice_model,
            "dynamic_pricing": dynamic_pricing,
            "overbooking": overbooking,
            "enable_holidays": enable_holidays,
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
            "time_sensitivity": time_sensitivity,
            "rm_optimization_frequency": rm_freq_hours,
            # Advanced Config
            "optimization_horizons": opt_horizons,
            "price_update_frequency_hours": price_update_freq,
            "demand_generation_method": demand_gen_method,
            "overbooking_method": overbooking_method,
            "overbooking_risk_tolerance": overbooking_risk,
            "include_buyup_down": include_buyup_down,
            "include_recapture": include_recapture,
            "personalization_enabled": personalization_enabled
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
                    st.session_state.sim_id = sim_id
                    st.session_state.sim_completed = True
                    break
                elif state == "failed":
                    st.error(f"Simulation Failed: {message}")
                    break
                
                time.sleep(1)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")

    if st.session_state.sim_completed and st.session_state.sim_id:
        display_results(st.session_state.sim_id)

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
        
    # Database Explorer
    st.divider()
    st.header("Database Explorer")
    
    try:
        # Fetch tables
        tables_res = requests.get(f"{API_URL}/simulations/{sim_id}/db/tables")
        if tables_res.status_code == 200:
            tables = tables_res.json().get("tables", [])
            
            if tables:
                selected_table = st.selectbox("Select Table", tables)
                
                # Fetch data
                data_res = requests.get(f"{API_URL}/simulations/{sim_id}/db/{selected_table}?limit=1000")
                if data_res.status_code == 200:
                    data = data_res.json()
                    df = pd.DataFrame(data)
                    st.dataframe(df)
                    
                    # Download CSV button
                    csv_url = f"{API_URL}/simulations/{sim_id}/db/{selected_table}/csv"
                    st.markdown(f"[Download {selected_table}.csv]({csv_url})")
                else:
                    st.error("Could not fetch table data")
            else:
                st.info("No tables found in database.")
        else:
            st.warning("Database not available for this simulation.")
    except Exception as e:
        st.error(f"Error connecting to database: {e}")

if __name__ == "__main__":
    main()
