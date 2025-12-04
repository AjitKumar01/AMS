"""
FastAPI server for the PyAirline RM Simulator.
"""

import sys
import uuid
import threading
import logging
import traceback
import sqlite3
from pathlib import Path
from datetime import date, timedelta
from typing import Dict, Any, List, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add parent directory to path to ensure imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.simulator import Simulator, SimulationConfig
from core.models import BookingClass
from demand.forecaster import DemandForecaster, ForecastMethod
from demand.generator import MultiStreamDemandGenerator
from inventory.network import NetworkOptimizer
from competition.market import Market

# Import helper functions from the example
from examples.competitive_simulation import (
    create_competitive_network,
    create_airlines,
    create_competitive_demand
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api.server")

DB_PATH = Path(__file__).parent.parent / "database" / "airports.db"

app = FastAPI(title="PyAirline RM API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Storage ---
# In a real app, use a database (Redis/Postgres)
simulations: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models ---

class SimulationRequest(BaseModel):
    start_date: date = Field(default=date(2025, 12, 1))
    end_date: date = Field(default=date(2025, 12, 7)) # Default to 1 week for speed
    rm_method: str = Field(default="EMSR-b")
    choice_model: str = Field(default="mnl")
    dynamic_pricing: bool = True
    overbooking: bool = True
    demand_multiplier: float = Field(default=1.0, ge=0.1, le=5.0)
    random_seed: int = 42
    
    # New fields
    selected_airlines: List[str] = Field(default=["AA", "UA", "DL"])
    demand_pattern: str = Field(default="default", description="default, high_business, high_leisure")
    single_flight_mode: bool = Field(default=False, description="Simulate only one route (JFK-LAX)")
    customer_currency: str = Field(default="USD")
    base_currency: str = Field(default="USD")
    
    # Advanced RM & Forecasting
    forecast_method: str = Field(default="pickup", description="pickup, additive_pickup, exponential_smoothing, historical_average, multiplicative_pickup, neural_network, xgboost, ensemble")
    rm_optimization_frequency: int = Field(default=24, description="Frequency of RM optimization in hours")
    
    # Advanced Configuration
    optimization_horizons: List[int] = Field(default=[30, 14, 7, 3, 1], description="Days before departure to run optimization")
    price_update_frequency_hours: float = Field(default=6.0, description="Frequency of price updates in hours")
    demand_generation_method: str = Field(default="poisson", description="poisson, stateful")
    overbooking_method: str = Field(default="critical_fractile", description="critical_fractile, risk_averse")
    overbooking_risk_tolerance: float = Field(default=0.05, description="Max probability of denied boarding")
    include_buyup_down: bool = Field(default=True, description="Allow customers to buy up/down")
    include_recapture: bool = Field(default=True, description="Allow recapture of spilled demand")
    personalization_enabled: bool = Field(default=False, description="Enable personalized offers")
    use_db: bool = Field(default=True, description="Enable database storage")
    
    # Detailed Demand Parameters
    demand_mean: float = Field(default=100.0, description="Mean daily demand per flight")
    demand_std: float = Field(default=20.0, description="Standard deviation of daily demand")
    business_proportion: float = Field(default=0.30, ge=0.0, le=1.0)
    business_wtp: float = Field(default=800.0, ge=0.0)
    leisure_wtp: float = Field(default=300.0, ge=0.0)
    
    # Custom Choice Parameters
    price_sensitivity: float = Field(default=-0.002)
    time_sensitivity: float = Field(default=-0.01)

class SimulationStatusResponse(BaseModel):
    simulation_id: str
    status: str  # pending, running, completed, failed
    progress: int
    message: str

class AirlineMetrics(BaseModel):
    airline_code: str
    airline_name: str
    revenue: float
    bookings: int
    load_factor: float
    avg_fare: float

class SimulationResultResponse(BaseModel):
    simulation_id: str
    status: str
    total_revenue: float
    total_bookings: int
    avg_load_factor: float
    airline_metrics: List[AirlineMetrics]
    exported_files: Dict[str, str] # Key: Description, Value: Download URL

class AirportInfo(BaseModel):
    iata_code: str
    name: str
    city: str
    country: str
    latitude: float
    longitude: float

# --- Background Task ---

def run_simulation_task(sim_id: str, request: SimulationRequest):
    """
    Runs the simulation in a background thread.
    """
    try:
        logger.info(f"Starting simulation {sim_id}")
        simulations[sim_id]["status"] = "running"
        simulations[sim_id]["progress"] = 10
        simulations[sim_id]["message"] = "Initializing network..."

        # Currency Rates (Simple hardcoded)
        rates = {
            "USD": 1.0,
            "EUR": 0.85,
            "GBP": 0.75,
            "JPY": 110.0,
            "AUD": 1.35,
            "CAD": 1.25
        }
        
        base_rate = rates.get(request.base_currency, 1.0)
        cust_rate = rates.get(request.customer_currency, 1.0)
        currency_rate = cust_rate / base_rate

        # 1. Setup Configuration
        # Base config - output_dir will be set per airline
        config = SimulationConfig(
            start_date=request.start_date,
            end_date=request.end_date,
            random_seed=request.random_seed,
            rm_method=request.rm_method,
            rm_optimization_frequency=request.rm_optimization_frequency,
            forecast_method=request.forecast_method,
            choice_model=request.choice_model,
            dynamic_pricing=request.dynamic_pricing,
            overbooking_enabled=request.overbooking,
            optimization_horizons=request.optimization_horizons,
            price_update_frequency_hours=request.price_update_frequency_hours,
            demand_generation_method=request.demand_generation_method,
            overbooking_method=request.overbooking_method,
            overbooking_risk_tolerance=request.overbooking_risk_tolerance,
            include_buyup_down=request.include_buyup_down,
            include_recapture=request.include_recapture,
            personalization_enabled=request.personalization_enabled,
            use_db=request.use_db,
            db_path=f"simulation_results/{sim_id}.db",
            progress_bar=False, # Disable tqdm
            export_csv=False,
            customer_currency=request.customer_currency,
            base_currency=request.base_currency,
            currency_rate=currency_rate
        )

        # 2. Create Network & Airlines
        network = create_competitive_network()
        airlines = create_airlines(network['routes'])
        
        # Filter airlines based on request
        if request.selected_airlines:
            airlines = {k: v for k, v in airlines.items() if k in request.selected_airlines}
            
        # Filter routes if single flight mode
        if request.single_flight_mode:
            # Keep only JFK-LAX
            network['routes'] = {k: v for k, v in network['routes'].items() if k == 'JFK-LAX'}
            # Filter airline schedules to match
            for code, airline in airlines.items():
                airline.schedules = [s for s in airline.schedules if s.route.origin.code == 'JFK' and s.route.destination.code == 'LAX']
        
        simulations[sim_id]["progress"] = 20
        simulations[sim_id]["message"] = "Setting up market and forecasters..."

        # 3. Setup Market
        market = Market(information_transparency=0.75)
        for airline in airlines.values():
            market.add_airline(airline)

        # 4. Setup Forecasters
        # Use requested method
        try:
            forecast_method_enum = ForecastMethod(request.forecast_method)
        except ValueError:
            logger.warning(f"Unknown forecast method {request.forecast_method}, defaulting to PICKUP")
            forecast_method_enum = ForecastMethod.PICKUP
            
        forecasters = {
            code: DemandForecaster(method=forecast_method_enum, track_accuracy=True, add_noise=True, noise_std=0.1)
            for code in airlines.keys()
        }

        simulations[sim_id]["progress"] = 30
        simulations[sim_id]["message"] = "Generating demand..."

        # 5. Generate Demand
        # We need to pass the filtered routes to create_competitive_demand, 
        # but create_competitive_demand takes the full routes dict from create_competitive_network.
        # However, it uses keys like 'JFK-LAX'.
        # If we filtered network['routes'], we might break create_competitive_demand if it expects all keys.
        # Let's check create_competitive_demand implementation.
        # It uses routes['JFK-LAX'] etc. So if we removed them from network['routes'], it will fail.
        
        # Solution: Generate ALL demand first, then filter streams based on available routes.
        full_network = create_competitive_network()
        demand_streams = create_competitive_demand(full_network['routes'])
        
        # Filter demand streams if single flight mode
        if request.single_flight_mode:
             demand_streams = [s for s in demand_streams if s.origin.code == 'JFK' and s.destination.code == 'LAX']
        
        # Apply demand pattern adjustments
        for stream in demand_streams:
            # Use explicit mean/std from request
            stream.mean_daily_demand = request.demand_mean * request.demand_multiplier
            stream.demand_std = request.demand_std
            
            # Apply user overrides if provided
            stream.business_proportion = request.business_proportion
            stream.business_wtp_mean = request.business_wtp
            stream.leisure_wtp_mean = request.leisure_wtp
            
            # Keep the pattern logic as modifiers if needed
            if request.demand_pattern == "high_business":
                stream.business_proportion = min(0.8, stream.business_proportion * 1.5)
                stream.business_wtp_mean *= 1.2
            elif request.demand_pattern == "high_leisure":
                stream.business_proportion = max(0.1, stream.business_proportion * 0.5)
                stream.leisure_wtp_mean *= 0.9
        
        demand_generator = MultiStreamDemandGenerator(demand_streams)
        all_requests = demand_generator.generate_all_requests(request.start_date, request.end_date)
        
        simulations[sim_id]["progress"] = 40
        simulations[sim_id]["message"] = f"Running simulation ({len(all_requests)} requests)..."

        # 6. Run Simulation Loop
        results_by_airline = {}
        total_airlines = len(airlines)
        all_exported_files = {}
        
        for i, (airline_code, airline) in enumerate(airlines.items()):
            simulations[sim_id]["message"] = f"Simulating {airline.name}..."
            
            # Set unique output directory for this airline
            airline_output_dir = f"simulation_results/{sim_id}/{airline_code}"
            config.output_dir = airline_output_dir
            
            simulator = Simulator(
                config=config,
                schedules=airline.schedules,
                routes=list(network['routes'].values()),
                airports=network['airports']
            )
            
            # Apply Custom Choice Model Parameters if needed
            if request.choice_model == "custom":
                from choice.models import MultinomialLogitModel, UtilityFunction
                # Create custom utility function
                utility_fn = UtilityFunction(
                    price_coef=request.price_sensitivity,
                    time_coef=request.time_sensitivity,
                    use_log_price=True # Keep this default for realism
                )
                simulator.choice_model = MultinomialLogitModel(utility_function=utility_fn)
                logger.info(f"Using Custom Choice Model: Price={request.price_sensitivity}, Time={request.time_sensitivity}")
            
            # Inject demand
            demand_generator.add_requests_to_event_queue(
                simulator.event_manager,
                all_requests
            )
            
            # Run
            results = simulator.run()
            results_by_airline[airline_code] = results
            
            # Collect exported files
            for name, path in results.exported_files.items():
                # Create a user-friendly key and a download URL
                key = f"{airline.name} - {name}"
                # Extract filename from path
                filename = Path(path).name
                url = f"/simulations/{sim_id}/files/{airline_code}/{filename}"
                all_exported_files[key] = url
            
            # Update progress
            current_progress = 40 + int(50 * (i + 1) / total_airlines)
            simulations[sim_id]["progress"] = current_progress

        simulations[sim_id]["progress"] = 90
        simulations[sim_id]["message"] = "Aggregating results..."

        # 7. Process Results
        airline_metrics = []
        total_rev = 0
        total_book = 0
        total_lf = 0
        
        for code, res in results_by_airline.items():
            metrics = AirlineMetrics(
                airline_code=code,
                airline_name=airlines[code].name,
                revenue=res.total_revenue,
                bookings=res.total_bookings,
                load_factor=res.load_factor * 100,
                avg_fare=res.total_revenue / res.total_bookings if res.total_bookings > 0 else 0
            )
            airline_metrics.append(metrics)
            total_rev += res.total_revenue
            total_book += res.total_bookings
            total_lf += res.load_factor

        avg_lf = (total_lf / len(airlines)) * 100 if airlines else 0

        result_data = SimulationResultResponse(
            simulation_id=sim_id,
            status="completed",
            total_revenue=total_rev,
            total_bookings=total_book,
            avg_load_factor=avg_lf,
            airline_metrics=airline_metrics,
            exported_files=all_exported_files
        )

        simulations[sim_id]["result"] = result_data
        simulations[sim_id]["status"] = "completed"
        simulations[sim_id]["progress"] = 100
        simulations[sim_id]["message"] = "Simulation completed successfully."
        
        logger.info(f"Simulation {sim_id} completed.")

    except Exception as e:
        logger.error(f"Simulation {sim_id} failed: {e}")
        traceback.print_exc()
        simulations[sim_id]["status"] = "failed"
        simulations[sim_id]["message"] = str(e)

# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/simulations", response_model=Dict[str, str])
def start_simulation(request: SimulationRequest, background_tasks: BackgroundTasks):
    sim_id = str(uuid.uuid4())
    simulations[sim_id] = {
        "id": sim_id,
        "status": "pending",
        "progress": 0,
        "message": "Queued",
        "request": request.dict(),
        "result": None
    }
    
    # Use threading to run it
    thread = threading.Thread(target=run_simulation_task, args=(sim_id, request))
    thread.start()
    
    return {"simulation_id": sim_id, "status": "pending"}

@app.get("/simulations/{sim_id}/status", response_model=SimulationStatusResponse)
def get_simulation_status(sim_id: str):
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = simulations[sim_id]
    return {
        "simulation_id": sim_id,
        "status": sim["status"],
        "progress": sim["progress"],
        "message": sim["message"]
    }

@app.get("/simulations/{sim_id}/results", response_model=SimulationResultResponse)
def get_simulation_results(sim_id: str):
    if sim_id not in simulations:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    sim = simulations[sim_id]
    if sim["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Simulation is {sim['status']}")
    
    return sim["result"]

@app.get("/simulations/{sim_id}/files/{airline_code}/{filename}")
def download_file(sim_id: str, airline_code: str, filename: str):
    """Download a specific simulation result file."""
    # Sanitize inputs to prevent directory traversal
    if ".." in airline_code or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid path")
        
    file_path = Path(f"simulation_results/{sim_id}/{airline_code}/{filename}")
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(file_path, filename=filename)

@app.get("/airports/{code}", response_model=AirportInfo)
def get_airport(code: str):
    """Get airport details by IATA code."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="Airport database not found")
        
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM airports WHERE iata_code = ?", (code.upper(),))
    row = cursor.fetchone()
    conn.close()
    
    if not row:
        raise HTTPException(status_code=404, detail="Airport not found")
        
    return AirportInfo(
        iata_code=row[0],
        name=row[1],
        city=row[2],
        country=row[3],
        latitude=row[4],
        longitude=row[5]
    )

@app.get("/simulations/{sim_id}/db/tables")
def get_db_tables(sim_id: str):
    """List tables in the simulation database."""
    db_path = Path(f"simulation_results/{sim_id}.db")
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return {"tables": tables}

@app.get("/simulations/{sim_id}/db/{table_name}")
def get_table_data(sim_id: str, table_name: str, limit: int = 100):
    """Get data from a table."""
    db_path = Path(f"simulation_results/{sim_id}.db")
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Use parameterized query for table name is not possible directly, 
        # but we can validate against list of tables first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        if table_name not in tables:
            raise HTTPException(status_code=404, detail="Table not found")
            
        cursor.execute(f"SELECT * FROM {table_name} LIMIT ?", (limit,))
        rows = [dict(row) for row in cursor.fetchall()]
        return rows
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

@app.get("/simulations/{sim_id}/db/{table_name}/csv")
def download_table_csv(sim_id: str, table_name: str):
    """Download table as CSV."""
    import csv
    import io
    from fastapi.responses import StreamingResponse
    
    db_path = Path(f"simulation_results/{sim_id}.db")
    if not db_path.exists():
        raise HTTPException(status_code=404, detail="Database not found")
        
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        if table_name not in tables:
            raise HTTPException(status_code=404, detail="Table not found")
            
        cursor.execute(f"SELECT * FROM {table_name}")
        
        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        if cursor.description:
            writer.writerow([d[0] for d in cursor.description])
            
        # Write rows
        for row in cursor:
            writer.writerow(row)
            
        output.seek(0)
        
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={table_name}.csv"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
