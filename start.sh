#!/bin/bash

# Airport Bottleneck Visualization - Start Script
# This script sets up and runs the Flask application

set -e  # Exit on any error

echo "🛩️  Airport Bottleneck Visualization - Starting Application"
echo "============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python 3 is installed
print_status "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Found Python $PYTHON_VERSION"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    print_error "pip3 is not installed. Please install pip3."
    exit 1
fi

print_success "Found pip3"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate
print_success "Virtual environment activated"

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_success "pip upgraded"

# Install dependencies
print_status "Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    print_success "Dependencies installed from requirements.txt"
else
    print_warning "requirements.txt not found, installing basic dependencies..."
    pip install Flask==2.3.3 requests==2.31.0 aiohttp==3.9.1 websockets==12.0
    print_success "Basic dependencies installed"
fi

# Check for additional dependencies that might be needed
print_status "Installing additional dependencies..."
pip install aiohttp websockets > /dev/null 2>&1 || true

# Create cache directory if it doesn't exist
if [ ! -d "cache" ]; then
    print_status "Creating cache directory..."
    mkdir -p cache
    print_success "Cache directory created"
fi

# Create templates directory if it doesn't exist
if [ ! -d "templates" ]; then
    print_warning "templates directory not found - this may cause issues"
fi

# Check if required files exist
print_status "Checking required files..."

REQUIRED_FILES=("app.py" "airport_data_fetcher.py" "world_data_processor.py" "client.py")
MISSING_FILES=()

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    print_warning "Some required files are missing: ${MISSING_FILES[*]}"
    print_warning "The application may not work properly without these files"
fi

# Set environment variables
export FLASK_APP=app.py
export FLASK_ENV=development
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Check if port 5001 is available
print_status "Checking if port 5001 is available..."
if lsof -Pi :5001 -sTCP:LISTEN -t >/dev/null 2>&1; then
    print_warning "Port 5001 is already in use. The application may fail to start."
    print_warning "You can kill the process using: lsof -ti:5001 | xargs kill -9"
fi

# Start the application
print_status "Starting Flask application..."
echo ""
echo "🚀 Application starting on http://localhost:5001"
echo "📊 Airport Bottleneck Visualization Dashboard"
echo "🛩️  Real-time aircraft monitoring enabled"
echo ""
echo "Press Ctrl+C to stop the application"
echo "============================================================="

# Run the Flask application
python3 app.py

print_status "Application stopped"
