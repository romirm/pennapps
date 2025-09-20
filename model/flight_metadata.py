"""
Flight Metadata Processor

Enriches ADS-B data with aircraft characteristics including passenger counts,
fuel capacity, and operational parameters for impact calculations.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np


class FlightMetadataProcessor:
    """
    Processes flight metadata to enrich ADS-B data with aircraft characteristics
    
    Provides:
    - Aircraft type database with passenger/fuel specifications
    - Passenger count estimation
    - Fuel consumption calculations
    - Operational phase analysis
    """
    
    def __init__(self):
        # Aircraft database with typical operational parameters
        self.aircraft_database = {
            # Commercial aircraft
            'B737': {
                'passengers': 150,
                'fuel_capacity': 6875,  # gallons
                'fuel_burn_idle': 150,  # gallons/hour
                'fuel_burn_cruise': 850,  # gallons/hour
                'max_range': 3500,  # nautical miles
                'category': 'narrow_body'
            },
            'A320': {
                'passengers': 180,
                'fuel_capacity': 6400,
                'fuel_burn_idle': 140,
                'fuel_burn_cruise': 800,
                'max_range': 3200,
                'category': 'narrow_body'
            },
            'B777': {
                'passengers': 350,
                'fuel_capacity': 47890,
                'fuel_burn_idle': 300,
                'fuel_burn_cruise': 2000,
                'max_range': 9700,
                'category': 'wide_body'
            },
            'A380': {
                'passengers': 550,
                'fuel_capacity': 84535,
                'fuel_burn_idle': 500,
                'fuel_burn_cruise': 3000,
                'max_range': 8000,
                'category': 'wide_body'
            },
            # Regional aircraft
            'CRJ9': {
                'passengers': 90,
                'fuel_capacity': 2600,
                'fuel_burn_idle': 80,
                'fuel_burn_cruise': 400,
                'max_range': 1500,
                'category': 'regional'
            },
            'E175': {
                'passengers': 80,
                'fuel_capacity': 2590,
                'fuel_burn_idle': 75,
                'fuel_burn_cruise': 380,
                'max_range': 1400,
                'category': 'regional'
            },
            # Cargo aircraft
            'B767F': {
                'passengers': 0,
                'cargo_value_estimate': 50000,  # USD
                'fuel_capacity': 24000,
                'fuel_burn_idle': 250,
                'fuel_burn_cruise': 1200,
                'max_range': 4000,
                'category': 'cargo'
            },
            'B747F': {
                'passengers': 0,
                'cargo_value_estimate': 100000,
                'fuel_capacity': 57000,
                'fuel_burn_idle': 400,
                'fuel_burn_cruise': 2000,
                'max_range': 4500,
                'category': 'cargo'
            }
        }
        
        # Load factor assumptions by route type
        self.load_factors = {
            'domestic': 0.85,
            'international': 0.80,
            'regional': 0.75,
            'cargo': 0.0
        }
        
        # Fuel prices (updated regularly in production)
        self.fuel_price_per_gallon = 3.50  # USD
        
    def estimate_passenger_count(self, flight_data: Dict) -> int:
        """
        Estimate passenger count based on aircraft type and route
        
        Args:
            flight_data: Flight data dictionary
            
        Returns:
            Estimated passenger count
        """
        aircraft_type = flight_data.get('aircraft_type', 'B737')
        route_type = flight_data.get('route_type', 'domestic')
        
        aircraft_specs = self.aircraft_database.get(aircraft_type, self.aircraft_database['B737'])
        base_passengers = aircraft_specs.get('passengers', 150)
        
        # Apply load factor
        load_factor = self.load_factors.get(route_type, 0.80)
        estimated_passengers = int(base_passengers * load_factor)
        
        # Add some randomness to simulate real-world variation
        variation = np.random.normal(1.0, 0.1)  # ±10% variation
        estimated_passengers = max(1, int(estimated_passengers * variation))
        
        return estimated_passengers
    
    def estimate_fuel_consumption(self, flight_data: Dict, delay_minutes: float) -> Dict[str, float]:
        """
        Estimate fuel consumption during delays
        
        Args:
            flight_data: Flight data dictionary
            delay_minutes: Delay duration in minutes
            
        Returns:
            Dictionary with fuel consumption estimates
        """
        aircraft_type = flight_data.get('aircraft_type', 'B737')
        phase = flight_data.get('phase', 'ground')
        
        aircraft_specs = self.aircraft_database.get(aircraft_type, self.aircraft_database['B737'])
        
        # Get appropriate fuel burn rate
        if phase == 'ground':
            fuel_burn_rate = aircraft_specs.get('fuel_burn_idle', 150)
        elif phase == 'holding_pattern':
            fuel_burn_rate = aircraft_specs.get('fuel_burn_idle', 150) * 1.5  # 1.5x for airborne holding
        elif phase == 'approach':
            fuel_burn_rate = aircraft_specs.get('fuel_burn_idle', 150) * 1.2  # Slightly higher for approach
        else:
            fuel_burn_rate = aircraft_specs.get('fuel_burn_idle', 150)
        
        # Calculate fuel consumption
        fuel_consumed = (fuel_burn_rate / 60) * delay_minutes  # gallons
        
        # Calculate cost and environmental impact
        fuel_cost = fuel_consumed * self.fuel_price_per_gallon
        co2_emissions = fuel_consumed * 21.1  # lbs CO2 per gallon
        
        return {
            'fuel_gallons': fuel_consumed,
            'fuel_cost_usd': fuel_cost,
            'co2_emissions_lbs': co2_emissions,
            'fuel_burn_rate_gph': fuel_burn_rate
        }
    
    def get_aircraft_characteristics(self, aircraft_type: str) -> Dict[str, any]:
        """
        Get aircraft characteristics from database
        
        Args:
            aircraft_type: Aircraft type code (e.g., 'B737')
            
        Returns:
            Dictionary of aircraft characteristics
        """
        return self.aircraft_database.get(aircraft_type, self.aircraft_database['B737'])
    
    def calculate_operational_impact(self, flight_data: Dict, delay_minutes: float) -> Dict[str, float]:
        """
        Calculate comprehensive operational impact
        
        Args:
            flight_data: Flight data dictionary
            delay_minutes: Delay duration in minutes
            
        Returns:
            Dictionary of operational impact metrics
        """
        # Passenger impact
        passengers = self.estimate_passenger_count(flight_data)
        
        # Fuel impact
        fuel_impact = self.estimate_fuel_consumption(flight_data, delay_minutes)
        
        # Calculate additional metrics
        aircraft_type = flight_data.get('aircraft_type', 'B737')
        aircraft_specs = self.get_aircraft_characteristics(aircraft_type)
        
        # Missed connections estimate (15% of passengers per hour of delay)
        missed_connections = passengers * 0.15 * (delay_minutes / 60)
        
        # Compensation cost estimate ($2.50 per passenger per minute)
        compensation_cost = passengers * delay_minutes * 2.50
        
        # Customer satisfaction impact (0-1 scale)
        satisfaction_impact = min(delay_minutes / 60 * 0.2, 1.0)
        
        # Cargo value impact (for cargo aircraft)
        cargo_impact = 0
        if aircraft_specs.get('category') == 'cargo':
            cargo_value = aircraft_specs.get('cargo_value_estimate', 0)
            cargo_impact = cargo_value * (delay_minutes / 60) * 0.1  # 10% value loss per hour
        
        return {
            'passengers_affected': passengers,
            'missed_connections': missed_connections,
            'compensation_cost_usd': compensation_cost,
            'customer_satisfaction_impact': satisfaction_impact,
            'cargo_value_impact_usd': cargo_impact,
            'fuel_gallons_wasted': fuel_impact['fuel_gallons'],
            'fuel_cost_usd': fuel_impact['fuel_cost_usd'],
            'co2_emissions_lbs': fuel_impact['co2_emissions_lbs'],
            'total_economic_impact_usd': compensation_cost + fuel_impact['fuel_cost_usd'] + cargo_impact
        }
    
    def process_flight_metadata(self, flight_data: Dict) -> Dict[str, any]:
        """
        Process and enrich flight metadata
        
        Args:
            flight_data: Raw flight data
            
        Returns:
            Enriched flight metadata
        """
        aircraft_type = flight_data.get('aircraft_type', 'B737')
        aircraft_specs = self.get_aircraft_characteristics(aircraft_type)
        
        # Estimate passenger count
        passengers = self.estimate_passenger_count(flight_data)
        
        # Determine route type based on flight data
        route_type = self._determine_route_type(flight_data)
        
        # Calculate operational parameters
        operational_params = {
            'aircraft_type': aircraft_type,
            'aircraft_category': aircraft_specs.get('category', 'unknown'),
            'estimated_passengers': passengers,
            'fuel_capacity_gallons': aircraft_specs.get('fuel_capacity', 0),
            'max_range_nm': aircraft_specs.get('max_range', 0),
            'route_type': route_type,
            'load_factor': self.load_factors.get(route_type, 0.80),
            'fuel_burn_idle_gph': aircraft_specs.get('fuel_burn_idle', 150),
            'fuel_burn_cruise_gph': aircraft_specs.get('fuel_burn_cruise', 800)
        }
        
        return operational_params
    
    def _determine_route_type(self, flight_data: Dict) -> str:
        """
        Determine route type based on flight data
        
        Args:
            flight_data: Flight data dictionary
            
        Returns:
            Route type string
        """
        aircraft_type = flight_data.get('aircraft_type', 'B737')
        aircraft_specs = self.get_aircraft_characteristics(aircraft_type)
        
        # Simple heuristic based on aircraft type
        if aircraft_specs.get('category') == 'cargo':
            return 'cargo'
        elif aircraft_specs.get('category') == 'regional':
            return 'regional'
        elif aircraft_specs.get('max_range', 0) > 5000:
            return 'international'
        else:
            return 'domestic'
    
    def update_fuel_prices(self, new_price_per_gallon: float):
        """
        Update fuel prices for cost calculations
        
        Args:
            new_price_per_gallon: New fuel price in USD per gallon
        """
        self.fuel_price_per_gallon = new_price_per_gallon
    
    def add_aircraft_type(self, aircraft_type: str, specifications: Dict):
        """
        Add new aircraft type to database
        
        Args:
            aircraft_type: Aircraft type code
            specifications: Aircraft specifications dictionary
        """
        self.aircraft_database[aircraft_type] = specifications
    
    def get_database_summary(self) -> Dict[str, int]:
        """
        Get summary of aircraft database
        
        Returns:
            Dictionary with database statistics
        """
        categories = {}
        for specs in self.aircraft_database.values():
            category = specs.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        
        return {
            'total_aircraft_types': len(self.aircraft_database),
            'categories': categories
        }
