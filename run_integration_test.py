import subprocess
import time
import requests
import sys
import os

API_BASE = "http://127.0.0.1:8000"

def wait_for_server(max_attempts=30):
    """Wait for server to be ready."""
    for i in range(max_attempts):
        try:
            response = requests.get(f"{API_BASE}/health", timeout=1)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(0.5)
    return False

def main():
    print("=" * 60)
    print("PyAirline RM API Integration Test")
    print("=" * 60)
    
    # Start server
    print("Starting API server...")
    server = subprocess.Popen(
        [sys.executable, "-m", "api.server"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    try:
        # Wait for server
        print("Waiting for server to be ready...")
        if not wait_for_server():
            print("❌ Server failed to start!")
            stdout, stderr = server.communicate()
            print(f"STDOUT:\n{stdout.decode()}")
            print(f"STDERR:\n{stderr.decode()}")
            return 1
        print("✅ Server is ready!\n")
        
        # Test Health
        print("Testing /health...")
        res = requests.get(f"{API_BASE}/health")
        assert res.status_code == 200
        print("✅ Health check passed\n")
        
        # Test Simulation
        print("Testing Simulation Workflow...")
        payload = {
            "start_date": "2025-12-01",
            "end_date": "2025-12-03", # Short duration
            "rm_method": "EMSR-b",
            "choice_model": "mnl",
            "dynamic_pricing": True,
            "overbooking": True,
            "demand_multiplier": 1.0,
            "selected_airlines": ["AA"], # Test filtering
            "single_flight_mode": True # Test single flight
        }
        
        res = requests.post(f"{API_BASE}/simulations", json=payload)
        assert res.status_code == 200
        data = res.json()
        sim_id = data["simulation_id"]
        print(f"✅ Simulation started: {sim_id}")
        
        # Poll
        print("Polling status...")
        for _ in range(60): # 60 seconds max
            res = requests.get(f"{API_BASE}/simulations/{sim_id}/status")
            status = res.json()
            print(f"   Progress: {status['progress']}% - {status['message']}")
            
            if status['status'] == 'completed':
                print("✅ Simulation completed!")
                break
            elif status['status'] == 'failed':
                print(f"❌ Simulation failed: {status['message']}")
                return 1
            
            time.sleep(1)
        else:
            print("❌ Timeout waiting for simulation")
            return 1
            
        # Get Results
        print("\nGetting Results...")
        res = requests.get(f"{API_BASE}/simulations/{sim_id}/results")
        assert res.status_code == 200
        results = res.json()
        print(f"✅ Results retrieved!")
        print(f"   Total Revenue: ${results['total_revenue']:,.2f}")
        print(f"   Total Bookings: {results['total_bookings']}")
        
        if 'exported_files' in results:
            print(f"   Exported Files: {len(results['exported_files'])}")
            # Verify one file download
            if results['exported_files']:
                first_file_url = list(results['exported_files'].values())[0]
                print(f"   Testing download: {first_file_url}")
                file_res = requests.get(f"{API_BASE}{first_file_url}")
                assert file_res.status_code == 200
                print("   ✅ File download successful")
        
        print("\nAll tests passed! 🎉")
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return 1
    finally:
        print("\nShutting down server...")
        server.terminate()
        server.wait()

if __name__ == "__main__":
    sys.exit(main())
