"""
Basic tests for GNN-KAN Airport Bottleneck Prediction System

Run with: python -m pytest model/tests.py
"""

import pytest
import torch
import numpy as np
from model.adsb_processor import ADSBDataProcessor, AircraftData
from model.flight_metadata import FlightMetadataProcessor
from model.impact_calculator import ImpactCalculator
from model.config import BOTTLENECK_CONFIG


class TestADSBProcessor:
    """Test ADS-B data processing functionality"""
    
    def test_aircraft_data_creation(self):
        """Test AircraftData dataclass creation"""
        aircraft = AircraftData(
            flight_id="TEST123",
            aircraft_type="B737",
            latitude=40.6413,
            longitude=-73.7781,
            altitude=500,
            heading=90,
            speed=150,
            phase="approach",
            timestamp="2024-01-01T12:00:00Z"
        )
        
        assert aircraft.flight_id == "TEST123"
        assert aircraft.aircraft_type == "B737"
        assert aircraft.latitude == 40.6413
        assert aircraft.phase == "approach"
    
    def test_processor_initialization(self):
        """Test ADSBDataProcessor initialization"""
        processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
        assert processor.spatial_resolution == 100
        assert processor.bottleneck_threshold == 0.6
    
    def test_airport_coordinates(self):
        """Test airport coordinate lookup"""
        processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
        
        # Test known airports
        jfk_coords = processor._get_airport_coordinates("KJFK")
        assert jfk_coords is not None
        assert jfk_coords[0] == 40.6413  # Latitude
        assert jfk_coords[1] == -73.7781  # Longitude
        
        # Test unknown airport
        unknown_coords = processor._get_airport_coordinates("UNKNOWN")
        assert unknown_coords is None
    
    def test_aircraft_phase_determination(self):
        """Test aircraft phase determination logic"""
        processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
        
        # Test approach phase
        approach_aircraft = {
            'alt_baro': 2000,
            'gs': 150
        }
        phase = processor._determine_aircraft_phase(approach_aircraft)
        assert phase == 'approach'
        
        # Test ground phase
        ground_aircraft = {
            'alt_baro': 50,
            'gs': 5
        }
        phase = processor._determine_aircraft_phase(ground_aircraft)
        assert phase == 'gate'
    
    def test_aircraft_type_encoding(self):
        """Test aircraft type encoding"""
        processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
        
        # Test known aircraft types
        b737_encoding = processor._encode_aircraft_type("B737")
        assert b737_encoding == 0.1
        
        a320_encoding = processor._encode_aircraft_type("A320")
        assert a320_encoding == 0.2
        
        # Test unknown aircraft type
        unknown_encoding = processor._encode_aircraft_type("UNKNOWN")
        assert unknown_encoding == 0.0


class TestFlightMetadataProcessor:
    """Test flight metadata processing functionality"""
    
    def test_processor_initialization(self):
        """Test FlightMetadataProcessor initialization"""
        processor = FlightMetadataProcessor()
        assert processor.fuel_price_per_gallon == 3.50
        assert processor.passenger_compensation_rate == 2.50
    
    def test_passenger_count_estimation(self):
        """Test passenger count estimation"""
        processor = FlightMetadataProcessor()
        
        # Test B737 domestic flight
        flight_data = {
            'aircraft_type': 'B737',
            'route_type': 'domestic'
        }
        passengers = processor.estimate_passenger_count(flight_data)
        assert passengers > 0
        assert passengers <= 150  # Should not exceed aircraft capacity
        
        # Test cargo aircraft
        cargo_flight_data = {
            'aircraft_type': 'B767F',
            'route_type': 'cargo'
        }
        passengers = processor.estimate_passenger_count(cargo_flight_data)
        assert passengers == 0  # Cargo aircraft have no passengers
    
    def test_fuel_consumption_estimation(self):
        """Test fuel consumption estimation"""
        processor = FlightMetadataProcessor()
        
        flight_data = {
            'aircraft_type': 'B737',
            'phase': 'ground'
        }
        
        fuel_consumption = processor.estimate_fuel_consumption(flight_data, 30)  # 30 minute delay
        
        assert fuel_consumption['fuel_gallons'] > 0
        assert fuel_consumption['fuel_cost_usd'] > 0
        assert fuel_consumption['co2_emissions_lbs'] > 0
        
        # Test that fuel consumption scales with delay time
        fuel_consumption_60min = processor.estimate_fuel_consumption(flight_data, 60)
        assert fuel_consumption_60min['fuel_gallons'] > fuel_consumption['fuel_gallons']
    
    def test_aircraft_characteristics(self):
        """Test aircraft characteristics lookup"""
        processor = FlightMetadataProcessor()
        
        b737_specs = processor.get_aircraft_characteristics("B737")
        assert b737_specs['passengers'] == 150
        assert b737_specs['fuel_capacity'] == 6875
        assert b737_specs['category'] == 'narrow_body'
        
        # Test unknown aircraft type
        unknown_specs = processor.get_aircraft_characteristics("UNKNOWN")
        assert unknown_specs['passengers'] == 150  # Should return B737 defaults
    
    def test_operational_impact_calculation(self):
        """Test operational impact calculation"""
        processor = FlightMetadataProcessor()
        
        flight_data = {
            'aircraft_type': 'B737',
            'phase': 'approach'
        }
        
        impact = processor.calculate_operational_impact(flight_data, 20)  # 20 minute delay
        
        assert impact['passengers_affected'] > 0
        assert impact['fuel_gallons_wasted'] > 0
        assert impact['total_economic_impact_usd'] > 0
        assert impact['customer_satisfaction_impact'] >= 0
        assert impact['customer_satisfaction_impact'] <= 1


class TestImpactCalculator:
    """Test impact calculation functionality"""
    
    def test_calculator_initialization(self):
        """Test ImpactCalculator initialization"""
        calculator = ImpactCalculator()
        assert calculator.passenger_compensation_rate == 2.50
        assert calculator.fuel_price_per_gallon == 3.50
        assert calculator.co2_per_gallon_fuel == 21.1
    
    def test_economic_impact_calculation(self):
        """Test economic impact calculation"""
        calculator = ImpactCalculator()
        
        bottleneck_data = {
            'type': 'runway_approach_queue',
            'severity': 0.7,
            'delay_minutes': 15
        }
        
        affected_flights = [
            {
                'aircraft_type': 'B737',
                'estimated_passengers': 150,
                'phase': 'approach'
            },
            {
                'aircraft_type': 'A320',
                'estimated_passengers': 180,
                'phase': 'approach'
            }
        ]
        
        economic_impact = calculator.calculate_economic_impact(bottleneck_data, affected_flights)
        
        assert economic_impact['total_economic_impact_usd'] > 0
        assert economic_impact['cost_per_minute'] > 0
        assert economic_impact['cost_per_affected_passenger'] > 0
        
        # Check that direct costs are calculated
        assert economic_impact['direct_costs']['fuel_waste_cost'] > 0
        assert economic_impact['direct_costs']['passenger_compensation'] > 0
    
    def test_environmental_impact_calculation(self):
        """Test environmental impact calculation"""
        calculator = ImpactCalculator()
        
        bottleneck_data = {
            'type': 'runway_departure_queue',
            'severity': 0.5,
            'delay_minutes': 10
        }
        
        affected_flights = [
            {
                'aircraft_type': 'B737',
                'estimated_passengers': 150,
                'phase': 'ground'
            }
        ]
        
        environmental_impact = calculator.calculate_environmental_impact(bottleneck_data, affected_flights)
        
        assert environmental_impact['fuel_waste_gallons'] > 0
        assert environmental_impact['co2_emissions_lbs'] > 0
        assert environmental_impact['co2_emissions_tons'] > 0
        assert environmental_impact['environmental_cost_usd'] > 0
    
    def test_passenger_impact_calculation(self):
        """Test passenger impact calculation"""
        calculator = ImpactCalculator()
        
        affected_flights = [
            {
                'aircraft_type': 'B737',
                'estimated_passengers': 150,
                'phase': 'approach'
            },
            {
                'aircraft_type': 'A320',
                'estimated_passengers': 180,
                'phase': 'approach'
            }
        ]
        
        passenger_impact = calculator.calculate_passenger_impact(affected_flights, 20)
        
        assert passenger_impact['passengers_affected'] == 330  # 150 + 180
        assert passenger_impact['missed_connections'] > 0
        assert passenger_impact['customer_satisfaction_impact'] >= 0
        assert passenger_impact['customer_satisfaction_impact'] <= 1
        assert passenger_impact['compensation_cost'] > 0
    
    def test_mitigation_cost_benefit(self):
        """Test mitigation cost-benefit analysis"""
        calculator = ImpactCalculator()
        
        bottleneck_data = {
            'type': 'runway_approach_queue',
            'severity': 0.8,
            'delay_minutes': 25
        }
        
        mitigation_options = [
            {
                'action': 'Increase runway separation',
                'implementation_cost': 0,
                'estimated_effectiveness': 0.8,
                'implementation_time': 5.0
            },
            {
                'action': 'Divert aircraft to alternate runways',
                'implementation_cost': 5000,
                'estimated_effectiveness': 0.9,
                'implementation_time': 10.0
            }
        ]
        
        cost_benefit_analysis = calculator.calculate_mitigation_cost_benefit(
            bottleneck_data, mitigation_options
        )
        
        assert len(cost_benefit_analysis) == 2
        
        for analysis in cost_benefit_analysis:
            assert 'mitigation_name' in analysis
            assert 'implementation_cost' in analysis
            assert 'avoided_costs' in analysis
            assert 'net_benefit' in analysis
            assert 'roi_percentage' in analysis
            assert 'recommendation' in analysis


class TestModelIntegration:
    """Test model integration and data flow"""
    
    def test_config_loading(self):
        """Test configuration loading"""
        assert BOTTLENECK_CONFIG['bottleneck_detection_threshold'] == 0.6
        assert BOTTLENECK_CONFIG['gnn_hidden_dim'] == 64
        assert BOTTLENECK_CONFIG['fuel_price_per_gallon'] == 3.50
    
    def test_torch_availability(self):
        """Test PyTorch availability"""
        assert torch is not None
        assert hasattr(torch, 'tensor')
        assert hasattr(torch, 'nn')
    
    def test_data_flow(self):
        """Test basic data flow through components"""
        # Create sample data
        processor = ADSBDataProcessor(BOTTLENECK_CONFIG)
        metadata_processor = FlightMetadataProcessor()
        
        sample_adsb = {
            "aircraft": [
                {
                    "flight": "TEST123",
                    "t": "B737",
                    "lat": 40.6413,
                    "lon": -73.7781,
                    "alt_baro": 500,
                    "track": 90,
                    "gs": 150,
                    "timestamp": "2024-01-01T12:00:00Z"
                }
            ]
        }
        
        # Process data
        aircraft_list = processor.filter_airport_operations(sample_adsb, "KJFK")
        assert len(aircraft_list) == 1
        
        aircraft = aircraft_list[0]
        assert aircraft.flight_id == "TEST123"
        assert aircraft.aircraft_type == "B737"
        
        # Process metadata
        flight_data = {
            'aircraft_type': aircraft.aircraft_type,
            'phase': aircraft.phase
        }
        
        passengers = metadata_processor.estimate_passenger_count(flight_data)
        assert passengers > 0
        
        fuel_consumption = metadata_processor.estimate_fuel_consumption(flight_data, 15)
        assert fuel_consumption['fuel_gallons'] > 0


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
