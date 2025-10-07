#!/bin/bash

# Cryptocurrency Forecasting System - Launch Script
# This script starts all components of the system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  CRYPTOCURRENCY FORECASTING SYSTEM - LAUNCHER                     ║"
echo "║  Starting API, Dashboard, and Testing Suite                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC}  $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC}  $1"
}

#!/bin/bash

# Cryptocurrency Forecasting System - Launch Script
# This script starts all components of the system

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  CRYPTOCURRENCY FORECASTING SYSTEM - LAUNCHER                     ║"
echo "║  Starting API, Dashboard, and Testing Suite                       ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to print status
print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠️${NC}  $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC}  $1"
}

# Check Python installation
echo ""
print_info "Checking system requirements..."

if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
print_status "Python $PYTHON_VERSION found"

# Check if virtual environment should be created
if [ ! -d "venv" ]; then
    print_warning "Virtual environment not found. Creating one..."
    python3 -m venv venv
    print_status "Virtual environment created"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate || . venv/Scripts/activate 2>/dev/null
print_status "Virtual environment activated"

# Check and install dependencies
print_info "Checking dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -q -r requirements.txt
    print_status "Dependencies installed/updated"
else
    print_warning "requirements.txt not found. Some dependencies may be missing."
fi

# Check data files
echo ""
print_info "Checking data files..."

if [ ! -f "data/raw/btc_data.csv" ] || [ ! -f "data/raw/eth_data.csv" ]; then
    print_warning "Data files not found. Would you like to fetch them now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Fetching cryptocurrency data..."
        python3 fetch_data.py
        if [ $? -eq 0 ]; then
            print_status "Data fetched successfully"
        else
            print_error "Data fetch failed. Please run 'python fetch_data.py' manually."
        fi
    else
        print_warning "Skipping data fetch. System may not work without data files."
    fi
else
    print_status "Data files found"
fi

# Create log directory
mkdir -p logs
print_status "Log directory ready"

# Start API Server
echo ""
print_info "Starting API Server..."
nohup python3 run_api.py > logs/api.log 2>&1 &
API_PID=$!
sleep 5

# Check if API started successfully
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    print_status "API Server started successfully (PID: $API_PID)"
    print_info "API available at: http://localhost:8000"
    print_info "API docs at: http://localhost:8000/docs"
else
    print_error "API Server failed to start. Check logs/api.log for details."
    print_info "Trying simplified API..."
    kill $API_PID 2>/dev/null
    nohup python3 run_api_simple.py > logs/api_simple.log 2>&1 &
    API_PID=$!
    sleep 5
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        print_status "Simplified API started (PID: $API_PID)"
    else
        print_error "Both API versions failed. Please check logs."
    fi
fi

# Start Dashboard
echo ""
print_info "Starting Interactive Dashboard..."
nohup python3 interactive_dashboard.py > logs/dashboard.log 2>&1 &
DASHBOARD_PID=$!
sleep 5

# Check if Dashboard started successfully
if curl -s http://localhost:8050 > /dev/null 2>&1; then
    print_status "Dashboard started successfully (PID: $DASHBOARD_PID)"
    print_info "Dashboard available at: http://localhost:8050"
else
    print_warning "Dashboard may not have started. Check logs/dashboard.log"
fi

# Save PIDs for cleanup
echo $API_PID > logs/api.pid
echo $DASHBOARD_PID > logs/dashboard.pid

# Display system status
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}SYSTEM STARTED SUCCESSFULLY!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Access Points:${NC}"
echo -e "  🌐 Dashboard:       ${BLUE}http://localhost:8050${NC}"
echo -e "  📡 API:             ${BLUE}http://localhost:8000${NC}"
echo -e "  📖 API Docs:        ${BLUE}http://localhost:8000/docs${NC}"
echo -e "  🔧 API ReDoc:       ${BLUE}http://localhost:8000/redoc${NC}"
echo ""
echo -e "${BOLD}Process IDs:${NC}"
echo -e "  API Server:         ${GREEN}$API_PID${NC}"
echo -e "  Dashboard:          ${GREEN}$DASHBOARD_PID${NC}"
echo ""
echo -e "${BOLD}Log Files:${NC}"
echo -e "  API Log:            ${BLUE}logs/api.log${NC}"
echo -e "  Dashboard Log:      ${BLUE}logs/dashboard.log${NC}"
echo ""
echo -e "${BOLD}Quick Commands:${NC}"
echo -e "  View API log:       ${YELLOW}tail -f logs/api.log${NC}"
echo -e "  View Dashboard log: ${YELLOW}tail -f logs/dashboard.log${NC}"
echo -e "  Stop system:        ${YELLOW}./stop_system.sh${NC}"
echo -e "  Run tests:          ${YELLOW}python3 test_complete_system.py${NC}"
echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════════════${NC}"
echo ""

# Ask if user wants to run tests
print_info "Would you like to run the comprehensive test suite? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo ""
    print_info "Running comprehensive tests..."
    sleep 3  # Give services time to fully start
    python3 test_complete_system.py
else
    print_info "Skipping tests. You can run them later with: python3 test_complete_system.py"
fi

echo ""
print_status "System is ready! Press Ctrl+C to stop monitoring or use './stop_system.sh' to stop services."
echo ""

# Monitor logs (optional)
print_info "Monitoring logs... (Press Ctrl+C to exit)"
tail -f logs/api.log logs/dashboard.log 2>/dev/null