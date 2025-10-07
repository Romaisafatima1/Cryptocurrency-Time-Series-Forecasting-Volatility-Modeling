#!/bin/bash

# Cryptocurrency Forecasting System - Stop Script
# This script stops all running components

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║  CRYPTOCURRENCY FORECASTING SYSTEM - SHUTDOWN                     ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

print_status() {
    echo -e "${GREEN}✅${NC} $1"
}

print_error() {
    echo -e "${RED}❌${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ️${NC}  $1"
}

# Stop processes from PID files
if [ -f "logs/api.pid" ]; then
    API_PID=$(cat logs/api.pid)
    if ps -p $API_PID > /dev/null 2>&1; then
        kill $API_PID
        print_status "API Server stopped (PID: $API_PID)"
    else
        print_info "API Server not running"
    fi
    rm logs/api.pid
else
    print_info "API PID file not found"
fi

if [ -f "logs/dashboard.pid" ]; then
    DASHBOARD_PID=$(cat logs/dashboard.pid)
    if ps -p $DASHBOARD_PID > /dev/null 2>&1; then
        kill $DASHBOARD_PID
        print_status "Dashboard stopped (PID: $DASHBOARD_PID)"
    else
        print_info "Dashboard not running"
    fi
    rm logs/dashboard.pid
else
    print_info "Dashboard PID file not found"
fi

# Kill any remaining processes on ports 8000 and 8050
print_info "Cleaning up any remaining processes..."

# For port 8000 (API)
PORT_8000_PID=$(lsof -ti:8000 2>/dev/null)
if [ ! -z "$PORT_8000_PID" ]; then
    kill -9 $PORT_8000_PID
    print_status "Cleaned up process on port 8000"
fi

# For port 8050 (Dashboard)
PORT_8050_PID=$(lsof -ti:8050 2>/dev/null)
if [ ! -z "$PORT_8050_PID" ]; then
    kill -9 $PORT_8050_PID
    print_status "Cleaned up process on port 8050"
fi

echo ""
echo -e "${GREEN}System stopped successfully!${NC}"
echo ""
print_info "To restart the system, run: ./launch_system.sh"
echo ""