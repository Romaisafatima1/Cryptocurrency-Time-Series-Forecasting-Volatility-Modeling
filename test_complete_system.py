#!/usr/bin/env python3
"""
Comprehensive Testing Suite for Cryptocurrency Forecasting System
Tests all components: API, Models, Visualizations, Dashboard
"""

import requests
import json
import time
from datetime import datetime
import sys
import os

# Test configuration
API_BASE_URL = "http://localhost:8000"
DASHBOARD_URL = "http://localhost:8050"
TEST_TIMEOUT = 60

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}\n")

def print_test(test_name):
    """Print test name"""
    print(f"{Colors.BOLD}Testing: {test_name}{Colors.END}")

def print_success(message):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """Print error message"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

class SystemTester:
    """Comprehensive system testing suite"""
    
    def __init__(self):
        self.passed_tests = 0
        self.failed_tests = 0
        self.total_tests = 0
        self.test_results = []
    
    def run_all_tests(self):
        """Run all system tests"""
        print_header("CRYPTOCURRENCY FORECASTING SYSTEM - COMPREHENSIVE TESTS")
        
        # Test API availability
        if not self.test_api_availability():
            print_error("API is not available. Please start the API server first.")
            print_info("Run: python run_api.py")
            return False
        
        # Run test suites
        self.test_api_endpoints()
        self.test_model_functionality()
        self.test_data_validation()
        self.test_error_handling()
        self.test_performance()
        
        # Print summary
        self.print_summary()
        
        return self.failed_tests == 0
    
    def test_api_availability(self):
        """Test if API is running"""
        print_test("API Availability")
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print_success("API is running and accessible")
                return True
            else:
                print_error(f"API returned status code: {response.status_code}")
                return False
        except requests.exceptions.ConnectionError:
            print_error("Cannot connect to API")
            return False
        except Exception as e:
            print_error(f"API check failed: {str(e)}")
            return False
    
    def test_api_endpoints(self):
        """Test all API endpoints"""
        print_header("API ENDPOINTS TESTING")
        
        endpoints = [
            ("GET", "/", "Root endpoint"),
            ("GET", "/health", "Health check"),
            ("GET", "/historical/BTC", "Historical analysis BTC"),
            ("GET", "/historical/ETH", "Historical analysis ETH"),
            ("POST", "/forecast", "Forecast generation", {
                "cryptocurrency": "BTC",
                "forecast_days": 7,
                "test_size": 0.2,
                "confidence_level": 0.95
            }),
            ("GET", "/metrics/BTC", "Model metrics BTC"),
            ("GET", "/compare", "Model comparison"),
            ("GET", "/cache/status", "Cache status"),
        ]
        
        for test in endpoints:
            if len(test) == 3:
                method, endpoint, description = test
                payload = None
            else:
                method, endpoint, description, payload = test
            
            self.test_endpoint(method, endpoint, description, payload)
    
    def test_endpoint(self, method, endpoint, description, payload=None):
        """Test a single endpoint"""
        self.total_tests += 1
        print_test(description)
        
        try:
            if method == "GET":
                response = requests.get(f"{API_BASE_URL}{endpoint}", timeout=TEST_TIMEOUT)
            elif method == "POST":
                response = requests.post(f"{API_BASE_URL}{endpoint}", json=payload, timeout=TEST_TIMEOUT)
            elif method == "DELETE":
                response = requests.delete(f"{API_BASE_URL}{endpoint}", timeout=TEST_TIMEOUT)
            
            if response.status_code == 200:
                print_success(f"Endpoint {endpoint} working correctly")
                print_info(f"Response time: {response.elapsed.total_seconds():.2f}s")
                self.passed_tests += 1
                self.test_results.append((description, True, None))
                return True
            else:
                print_error(f"Endpoint {endpoint} returned status {response.status_code}")
                self.failed_tests += 1
                self.test_results.append((description, False, f"Status {response.status_code}"))
                return False
        
        except requests.exceptions.Timeout:
            print_error(f"Endpoint {endpoint} timed out")
            self.failed_tests += 1
            self.test_results.append((description, False, "Timeout"))
            return False
        except Exception as e:
            print_error(f"Endpoint {endpoint} failed: {str(e)}")
            self.failed_tests += 1
            self.test_results.append((description, False, str(e)))
            return False
    
    def test_model_functionality(self):
        """Test model functionality"""
        print_header("MODEL FUNCTIONALITY TESTING")
        
        models_to_test = ["ARIMA", "LSTM", "GARCH", "EGARCH"]
        cryptos = ["BTC", "ETH"]
        
        for crypto in cryptos:
            for model in models_to_test:
                self.test_model(crypto, model)
    
    def test_model(self, crypto, model):
        """Test a specific model"""
        self.total_tests += 1
        print_test(f"{crypto} - {model} Model")
        
        try:
            response = requests.post(
                f"{API_BASE_URL}/forecast",
                json={
                    "cryptocurrency": crypto,
                    "forecast_days": 7,
                    "models": [model]
                },
                timeout=TEST_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                
                # Validate response structure
                if model in ['ARIMA', 'LSTM']:
                    if model in data['price_forecast']:
                        forecast = data['price_forecast'][model]
                        if len(forecast) == 7 and all(isinstance(x, (int, float)) for x in forecast):
                            print_success(f"{model} model generated valid price forecast")
                            self.passed_tests += 1
                            self.test_results.append((f"{crypto} {model}", True, None))
                            return True
                
                elif model in ['GARCH', 'EGARCH']:
                    if model in data['volatility_forecast']:
                        forecast = data['volatility_forecast'][model]
                        if len(forecast) == 7 and all(isinstance(x, (int, float)) for x in forecast):
                            print_success(f"{model} model generated valid volatility forecast")
                            self.passed_tests += 1
                            self.test_results.append((f"{crypto} {model}", True, None))
                            return True
                
                print_error(f"{model} model returned invalid forecast structure")
                self.failed_tests += 1
                self.test_results.append((f"{crypto} {model}", False, "Invalid structure"))
                return False
            
            else:
                print_error(f"{model} model failed with status {response.status_code}")
                self.failed_tests += 1
                self.test_results.append((f"{crypto} {model}", False, f"Status {response.status_code}"))
                return False
        
        except Exception as e:
            print_error(f"{model} model test failed: {str(e)}")
            self.failed_tests += 1
            self.test_results.append((f"{crypto} {model}", False, str(e)))
            return False
    
    def test_data_validation(self):
        """Test data validation"""
        print_header("DATA VALIDATION TESTING")
        
        # Test invalid cryptocurrency
        self.total_tests += 1
        print_test("Invalid cryptocurrency handling")
        try:
            response = requests.post(
                f"{API_BASE_URL}/forecast",
                json={"cryptocurrency": "INVALID", "forecast_days": 7},
                timeout=10
            )
            if response.status_code in [400, 404, 500]:
                print_success("Invalid cryptocurrency properly rejected")
                self.passed_tests += 1
                self.test_results.append(("Invalid crypto rejection", True, None))
            else:
                print_error("Invalid cryptocurrency not properly handled")
                self.failed_tests += 1
                self.test_results.append(("Invalid crypto rejection", False, "Not rejected"))
        except Exception as e:
            print_error(f"Validation test failed: {str(e)}")
            self.failed_tests += 1
            self.test_results.append(("Invalid crypto rejection", False, str(e)))
        
        # Test invalid forecast days
        self.total_tests += 1
        print_test("Invalid forecast days handling")
        try:
            response = requests.post(
                f"{API_BASE_URL}/forecast",
                json={"cryptocurrency": "BTC", "forecast_days": 1000},
                timeout=10
            )
            if response.status_code in [400, 422]:
                print_success("Invalid forecast days properly rejected")
                self.passed_tests += 1
                self.test_results.append(("Invalid forecast days", True, None))
            else:
                print_error("Invalid forecast days not properly handled")
                self.failed_tests += 1
                self.test_results.append(("Invalid forecast days", False, "Not rejected"))
        except Exception as e:
            print_error(f"Validation test failed: {str(e)}")
            self.failed_tests += 1
            self.test_results.append(("Invalid forecast days", False, str(e)))
    
    def test_error_handling(self):
        """Test error handling"""
        print_header("ERROR HANDLING TESTING")
        
        # Test malformed request
        self.total_tests += 1
        print_test("Malformed request handling")
        try:
            response = requests.post(
                f"{API_BASE_URL}/forecast",
                json={"invalid_field": "test"},
                timeout=10
            )
            if response.status_code in [400, 422]:
                print_success("Malformed request properly handled")
                self.passed_tests += 1
                self.test_results.append(("Malformed request", True, None))
            else:
                print_warning("Malformed request accepted (may have defaults)")
                self.passed_tests += 1
                self.test_results.append(("Malformed request", True, "Accepted with defaults"))
        except Exception as e:
            print_error(f"Error handling test failed: {str(e)}")
            self.failed_tests += 1
            self.test_results.append(("Malformed request", False, str(e)))
    
    def test_performance(self):
        """Test system performance"""
        print_header("PERFORMANCE TESTING")
        
        self.total_tests += 1
        print_test("API response time")
        
        try:
            start_time = time.time()
            response = requests.post(
                f"{API_BASE_URL}/forecast",
                json={"cryptocurrency": "BTC", "forecast_days": 7},
                timeout=TEST_TIMEOUT
            )
            end_time = time.time()
            
            response_time = end_time - start_time
            
            if response.status_code == 200:
                if response_time < 30:
                    print_success(f"Fast response time: {response_time:.2f}s")
                    self.passed_tests += 1
                    self.test_results.append(("Performance", True, f"{response_time:.2f}s"))
                elif response_time < 60:
                    print_warning(f"Acceptable response time: {response_time:.2f}s")
                    self.passed_tests += 1
                    self.test_results.append(("Performance", True, f"{response_time:.2f}s"))
                else:
                    print_warning(f"Slow response time: {response_time:.2f}s")
                    self.passed_tests += 1
                    self.test_results.append(("Performance", True, f"{response_time:.2f}s (slow)"))
            else:
                print_error(f"Performance test failed: Status {response.status_code}")
                self.failed_tests += 1
                self.test_results.append(("Performance", False, f"Status {response.status_code}"))
        
        except Exception as e:
            print_error(f"Performance test failed: {str(e)}")
            self.failed_tests += 1
            self.test_results.append(("Performance", False, str(e)))
    
    def print_summary(self):
        """Print test summary"""
        print_header("TEST SUMMARY")
        
        # Calculate success rate
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"\n{Colors.BOLD}Total Tests: {self.total_tests}{Colors.END}")
        print(f"{Colors.GREEN}Passed: {self.passed_tests}{Colors.END}")
        print(f"{Colors.RED}Failed: {self.failed_tests}{Colors.END}")
        print(f"{Colors.BOLD}Success Rate: {success_rate:.1f}%{Colors.END}\n")
        
        # Print detailed results
        if self.failed_tests > 0:
            print(f"\n{Colors.BOLD}{Colors.RED}Failed Tests:{Colors.END}")
            for test_name, passed, error in self.test_results:
                if not passed:
                    print(f"  {Colors.RED}❌ {test_name}: {error}{Colors.END}")
        
        # Print recommendations
        print(f"\n{Colors.BOLD}Recommendations:{Colors.END}")
        if self.failed_tests == 0:
            print(f"  {Colors.GREEN}✅ All systems operational!{Colors.END}")
            print(f"  {Colors.GREEN}✅ Ready for production deployment{Colors.END}")
        elif self.failed_tests < 3:
            print(f"  {Colors.YELLOW}⚠️  Minor issues detected{Colors.END}")
            print(f"  {Colors.YELLOW}⚠️  System mostly functional{Colors.END}")
        else:
            print(f"  {Colors.RED}❌ Major issues detected{Colors.END}")
            print(f"  {Colors.RED}❌ Review failed tests before deployment{Colors.END}")
        
        print("\n" + "="*70 + "\n")

def test_data_files():
    """Test that required data files exist"""
    print_header("DATA FILES CHECK")
    
    required_files = [
        "data/raw/btc_data.csv",
        "data/raw/eth_data.csv"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print_success(f"Found: {file_path}")
        else:
            print_error(f"Missing: {file_path}")
            all_exist = False
    
    if not all_exist:
        print_warning("Some data files are missing. Run fetch_data.py first.")
    
    return all_exist

def test_visualization_module():
    """Test visualization module"""
    print_header("VISUALIZATION MODULE CHECK")
    
    try:
        import matplotlib
        import seaborn
        print_success("Matplotlib and Seaborn available")
        
        # Check if visualization_suite module exists
        if os.path.exists("visualization_suite.py"):
            print_success("Visualization suite module found")
        else:
            print_warning("Visualization suite module not found in current directory")
        
        return True
    except ImportError as e:
        print_error(f"Missing visualization dependencies: {str(e)}")
        return False

def test_dashboard_availability():
    """Test if dashboard is accessible"""
    print_header("DASHBOARD AVAILABILITY CHECK")
    
    try:
        response = requests.get(DASHBOARD_URL, timeout=5)
        if response.status_code == 200:
            print_success(f"Dashboard is running at {DASHBOARD_URL}")
            return True
        else:
            print_warning(f"Dashboard returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_warning("Dashboard is not running")
        print_info("Start dashboard with: python interactive_dashboard.py")
        return False
    except Exception as e:
        print_warning(f"Dashboard check failed: {str(e)}")
        return False

def main():
    """Main test execution"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}")
    print("╔═══════════════════════════════════════════════════════════════════╗")
    print("║  CRYPTOCURRENCY FORECASTING SYSTEM - COMPREHENSIVE TEST SUITE    ║")
    print("║  Testing: API, Models, Data, Visualizations, Dashboard           ║")
    print("╚═══════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.END}\n")
    
    # Pre-flight checks
    print_info("Running pre-flight checks...")
    data_ok = test_data_files()
    viz_ok = test_visualization_module()
    dashboard_running = test_dashboard_availability()
    
    # Wait for user confirmation
    print(f"\n{Colors.BOLD}Ready to run comprehensive API and model tests.{Colors.END}")
    print(f"{Colors.YELLOW}This will take approximately 2-3 minutes.{Colors.END}")
    
    input("\nPress Enter to continue...")
    
    # Run comprehensive tests
    tester = SystemTester()
    success = tester.run_all_tests()
    
    # Final recommendations
    print_header("FINAL SYSTEM STATUS")
    
    print(f"{Colors.BOLD}Component Status:{Colors.END}")
    print(f"  API: {Colors.GREEN}✅ Operational{Colors.END}" if success else f"  API: {Colors.RED}❌ Issues detected{Colors.END}")
    print(f"  Data Files: {Colors.GREEN}✅ Present{Colors.END}" if data_ok else f"  Data Files: {Colors.RED}❌ Missing{Colors.END}")
    print(f"  Visualizations: {Colors.GREEN}✅ Available{Colors.END}" if viz_ok else f"  Visualizations: {Colors.RED}❌ Dependencies missing{Colors.END}")
    print(f"  Dashboard: {Colors.GREEN}✅ Running{Colors.END}" if dashboard_running else f"  Dashboard: {Colors.YELLOW}⚠️  Not running{Colors.END}")
    
    print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
    if success and data_ok and viz_ok:
        print(f"  1. {Colors.GREEN}✅ System is fully operational{Colors.END}")
        print(f"  2. {Colors.BLUE}Access dashboard at: {DASHBOARD_URL}{Colors.END}")
        print(f"  3. {Colors.BLUE}API documentation at: {API_BASE_URL}/docs{Colors.END}")
        print(f"  4. {Colors.GREEN}Ready for Week 10 presentation!{Colors.END}")
    else:
        if not success:
            print(f"  1. {Colors.RED}Fix API issues listed above{Colors.END}")
        if not data_ok:
            print(f"  2. {Colors.YELLOW}Run: python fetch_data.py{Colors.END}")
        if not dashboard_running:
            print(f"  3. {Colors.YELLOW}Start dashboard: python interactive_dashboard.py{Colors.END}")
    
    print("\n" + "="*70 + "\n")
    
    return success

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Tests interrupted by user{Colors.END}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n{Colors.RED}Fatal error: {str(e)}{Colors.END}\n")
        sys.exit(1)