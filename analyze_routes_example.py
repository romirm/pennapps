#!/usr/bin/env python3
"""
Example usage of the analyze_routes function.
"""

from client import PlaneMonitor

def main():
    """Demonstrate the analyze_routes function."""
    
    # Create PlaneMonitor instance
    monitor = PlaneMonitor()
    
    # Example flight numbers (these may return errors since they're not real active flights)
    flight_numbers = ["DAL1126", "JBU456", "AAL789"]
    
    print("ADSB Route Analysis Example")
    print("=" * 40)
    print(f"Analyzing routes for: {flight_numbers}")
    print()
    
    # Call the analyze_routes function
    destinations = monitor.analyze_routes(flight_numbers)
    
    print("Results:")
    for flight, destination in destinations.items():
        print(f"  {flight}: {destination}")
    
    print()
    print("Note: API errors are expected for test flight numbers.")
    print("The function will work with real, active flight numbers.")
    print()
    print("Example usage in your code:")
    print("```python")
    print("from client import PlaneMonitor")
    print("monitor = PlaneMonitor()")
    print("destinations = monitor.analyze_routes(['DAL123', 'JBU456', 'AAL789'])")
    print("```")

if __name__ == "__main__":
    main()
