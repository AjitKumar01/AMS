#!/bin/bash
# Quick Docker build and run script for PyAirline RM

set -e  # Exit on error

echo "=================================================="
echo "PyAirline RM - Docker Quick Start"
echo "=================================================="
echo ""

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install Docker from https://www.docker.com/get-started"
    exit 1
fi

echo -e "${BLUE}Step 1:${NC} Building Docker image..."
docker build -t pyairline-rm:latest .

echo ""
echo -e "${GREEN}✓${NC} Image built successfully"
echo ""

echo -e "${BLUE}Step 2:${NC} Running feature tests..."
docker run --rm pyairline-rm:latest python test_features.py

echo ""
echo -e "${GREEN}✓${NC} Tests passed"
echo ""

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo -e "${BLUE}Step 3:${NC} Running competitive simulation..."
echo "(This may take ~2 minutes)"
echo ""
docker run --rm -v "$(pwd)/outputs:/app/outputs" pyairline-rm:latest

echo ""
echo "=================================================="
echo -e "${GREEN}✓ All done!${NC}"
echo "=================================================="
echo ""
echo "Results have been saved to: ./outputs/"
echo ""
echo "Next steps:"
echo "  • Run basic example: docker run --rm pyairline-rm:latest python examples/basic_example.py"
echo "  • Interactive shell: docker run --rm -it pyairline-rm:latest python"
echo "  • Development mode: docker-compose --profile dev up -d pyairline-dev"
echo ""
echo "See DOCKER.md for more options"
