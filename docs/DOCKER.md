# PyAirline RM - Docker Guide

## Quick Start

### Build and Run (Competitive Simulation)

```bash
# Build the image
docker build -t pyairline-rm .

# Run competitive simulation
docker run --rm -v $(pwd)/outputs:/app/outputs pyairline-rm

# Run basic example
docker run --rm -v $(pwd)/outputs:/app/outputs pyairline-rm python examples/basic_example.py

# Run feature test
docker run --rm pyairline-rm python test_features.py
```

### Using Docker Compose

```bash
# Run competitive simulation (default)
docker-compose up

# Run basic example
docker-compose --profile basic up pyairline-basic

# Interactive development environment
docker-compose --profile dev up pyairline-dev

# When dashboard is implemented
docker-compose --profile dashboard up pyairline-dashboard
```

## Docker Images

### Production Image (`Dockerfile`)

**Purpose**: Run simulations in production  
**Size**: ~800 MB  
**Base**: python:3.11-slim  

**Features:**
- Minimal dependencies
- Pre-runs tests during build
- Optimized for speed
- Includes all required packages

**Build:**
```bash
docker build -t pyairline-rm:latest .
```

**Run:**
```bash
docker run --rm pyairline-rm
```

### Development Image (`Dockerfile.dev`)

**Purpose**: Interactive development  
**Size**: ~1.2 GB  
**Base**: python:3.11-slim  

**Features:**
- Development tools (ipython, jupyter, pytest)
- Code formatting tools (black, flake8)
- Interactive shell
- Volume mounting for live edits

**Build:**
```bash
docker build -f Dockerfile.dev -t pyairline-rm:dev .
```

**Run:**
```bash
docker run --rm -it -v $(pwd):/app pyairline-rm:dev
```

## Common Use Cases

### 1. Run Competitive Simulation

```bash
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  pyairline-rm \
  python examples/competitive_simulation.py
```

**Output**: Results saved to `./outputs/`

### 2. Run Basic Example

```bash
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  pyairline-rm \
  python examples/basic_example.py
```

### 3. Interactive Python Shell

```bash
docker run --rm -it pyairline-rm python
```

Then in Python:
```python
from competition.airline import Airline, CompetitiveStrategy
from competition.market import Market

aa = Airline("AA", "American", CompetitiveStrategy.AGGRESSIVE)
print(aa)
```

### 4. Development Environment

```bash
# Start development container
docker-compose --profile dev up -d pyairline-dev

# Attach to it
docker exec -it pyairline-dev bash

# Inside container
python examples/competitive_simulation.py
python test_features.py
ipython  # Interactive Python
```

### 5. Run Custom Python Script

```bash
# Create your script
echo 'print("Hello from PyAirline RM!")' > my_script.py

# Run it
docker run --rm \
  -v $(pwd)/my_script.py:/app/my_script.py \
  pyairline-rm \
  python my_script.py
```

### 6. Export Results to CSV

```bash
docker run --rm \
  -v $(pwd)/outputs:/app/outputs \
  pyairline-rm \
  python -c "
from examples.competitive_simulation import run_competitive_simulation
import pandas as pd

results = run_competitive_simulation()

# Export to CSV
for airline_code, result in results['results'].items():
    df = pd.DataFrame([{
        'airline': airline_code,
        'revenue': result.total_revenue,
        'bookings': result.total_bookings,
        'load_factor': result.load_factor
    }])
    df.to_csv(f'/app/outputs/{airline_code}_results.csv', index=False)
"
```

## Docker Compose Profiles

### Default Profile (Competitive Simulation)

```bash
docker-compose up
```

Runs the full competitive simulation with 3 airlines.

### Basic Profile

```bash
docker-compose --profile basic up
```

Runs the basic single-airline example.

### Development Profile

```bash
docker-compose --profile dev up
```

Starts interactive development environment with:
- Live code mounting
- Development tools installed
- Interactive shell

### Dashboard Profile (Future)

```bash
docker-compose --profile dashboard up
```

Starts the web dashboard (when implemented).

## Volume Mounting

### Output Directory

```bash
-v $(pwd)/outputs:/app/outputs
```

Saves simulation results to your local `outputs/` directory.

### Live Code Development

```bash
-v $(pwd):/app
```

Mount entire codebase for live editing (development only).

### Custom Configuration

```bash
-v $(pwd)/my_config.yaml:/app/config.yaml
```

Mount custom configuration file.

## Environment Variables

### Python Configuration

```bash
docker run --rm \
  -e PYTHONUNBUFFERED=1 \
  -e PYTHONDONTWRITEBYTECODE=1 \
  pyairline-rm
```

### Simulation Parameters (Example)

```bash
docker run --rm \
  -e SIMULATION_DAYS=60 \
  -e NUM_AIRLINES=5 \
  pyairline-rm \
  python my_custom_simulation.py
```

## Multi-Stage Builds (Advanced)

For smaller production images, you can use multi-stage builds:

```dockerfile
# Builder stage
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "examples/competitive_simulation.py"]
```

## Troubleshooting

### Issue: Container exits immediately

**Solution**: Check logs
```bash
docker logs pyairline-rm
```

### Issue: Permission denied on outputs

**Solution**: Create outputs directory first
```bash
mkdir -p outputs
chmod 777 outputs
```

### Issue: Out of memory

**Solution**: Increase Docker memory limit
```bash
docker run --rm -m 4g pyairline-rm
```

### Issue: Slow build

**Solution**: Use build cache
```bash
docker build --cache-from pyairline-rm:latest -t pyairline-rm .
```

### Issue: Can't find module

**Solution**: Rebuild without cache
```bash
docker build --no-cache -t pyairline-rm .
```

## Performance Tips

### 1. Layer Caching

Order Dockerfile commands from least to most frequently changed:
- System packages
- Python requirements
- Application code

### 2. Multi-Core Simulation

```bash
docker run --rm --cpus=4 pyairline-rm
```

### 3. Memory Allocation

```bash
docker run --rm -m 8g pyairline-rm
```

### 4. Parallel Runs

```bash
# Run multiple simulations in parallel
for i in {1..5}; do
  docker run --rm -d \
    -v $(pwd)/outputs:/app/outputs \
    --name sim-$i \
    pyairline-rm \
    python examples/competitive_simulation.py
done
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Run Simulations

on: [push]

jobs:
  simulate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build Docker image
        run: docker build -t pyairline-rm .
      
      - name: Run tests
        run: docker run --rm pyairline-rm python test_features.py
      
      - name: Run simulation
        run: docker run --rm -v $(pwd)/outputs:/app/outputs pyairline-rm
      
      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: simulation-results
          path: outputs/
```

## Best Practices

1. **Always mount outputs volume** for persistent results
2. **Use specific tags** instead of `:latest` in production
3. **Set memory limits** for large simulations
4. **Use `.dockerignore`** to reduce build context
5. **Run tests during build** to catch issues early
6. **Use multi-stage builds** for smaller images
7. **Clean up** stopped containers regularly

## Image Size Optimization

Current sizes:
- Production: ~800 MB
- Development: ~1.2 GB

To reduce:
```dockerfile
# Use alpine (smaller but may have compatibility issues)
FROM python:3.11-alpine

# Or use distroless
FROM gcr.io/distroless/python3
```

## Container Registry

### Push to Docker Hub

```bash
# Tag
docker tag pyairline-rm:latest username/pyairline-rm:latest

# Push
docker push username/pyairline-rm:latest
```

### Pull and Run

```bash
docker pull username/pyairline-rm:latest
docker run --rm username/pyairline-rm:latest
```

## Cleanup

### Remove containers
```bash
docker-compose down
```

### Remove images
```bash
docker rmi pyairline-rm:latest
docker rmi pyairline-rm:dev
```

### Remove all
```bash
docker system prune -a
```

## Summary

**Quick Commands:**
```bash
# Build
docker build -t pyairline-rm .

# Run competitive simulation
docker run --rm -v $(pwd)/outputs:/app/outputs pyairline-rm

# Run basic example  
docker run --rm pyairline-rm python examples/basic_example.py

# Development
docker-compose --profile dev up -d pyairline-dev
docker exec -it pyairline-dev bash

# Cleanup
docker-compose down
```

**The Docker setup is production-ready and provides multiple deployment options.**
