# PyAirline RM - Production Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY . .

# Install the package
RUN pip install -e .

# Create directory for outputs
RUN mkdir -p /app/outputs

# Run tests to verify installation
RUN python test_features.py

# Default command - run the competitive simulation
CMD ["python", "examples/competitive_simulation.py"]

# Expose port for future dashboard (optional)
EXPOSE 8050

# Health check (optional)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pyairline_rm; print('OK')" || exit 1
