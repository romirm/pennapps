"""
Impact Calculator

Calculates comprehensive impact metrics for bottlenecks including economic,
environmental, and operational impacts.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta


class ImpactCalculator:
    """
    Calculates comprehensive impact metrics for airport bottlenecks
    
    Provides:
    - Economic impact analysis
    - Environmental impact assessment
    - Operational efficiency metrics
    - Cost-benefit analysis for mitigations
    """
    
    def __init__(self):
        # Economic parameters
        self.passenger_compensation_rate = 2.50  # USD per passenger per minute
        self.fuel_price_per_gallon = 3.50  # USD
        self.co2_cost_per_ton = 50.0  # USD (carbon pricing)
        self.airline_reputation_cost = 1000.0  # USD per incident
        
        # Environmental conversion factors
        self.co2_per_gallon_fuel = 21.1  # lbs CO2 per gallon
        self.co2_per_ton = 2204.62  # lbs per ton
        
        # Operational efficiency metrics
        self.on_time_performance_weight = 0.3
        self.customer_satisfaction_weight = 0.4
        self.operational_cost_weight = 0.3
        
    def calculate_economic_impact(self, bottleneck_data: Dict, 
                                affected_flights: List[Dict]) -> Dict[str, float]:
        """
        Calculate economic impact of bottlenecks
        
        Args:
            bottleneck_data: Bottleneck analysis data
            affected_flights: List of affected flight data
            
        Returns:
            Dictionary of economic impact metrics
        """
        delay_minutes = bottleneck_data.get('delay_minutes', 0)
        bottleneck_type = bottleneck_data.get('type', 'unknown')
        
        # Direct costs
        direct_costs = {
            'fuel_waste_cost': 0,
            'passenger_compensation': 0,
            'operational_delays': 0,
            'cargo_value_loss': 0
        }
        
        # Calculate costs per flight
        for flight in affected_flights:
            passengers = flight.get('estimated_passengers', 0)
            aircraft_type = flight.get('aircraft_type', 'B737')
            
            # Fuel waste cost
            fuel_consumption = self._calculate_flight_fuel_consumption(flight, delay_minutes)
            direct_costs['fuel_waste_cost'] += fuel_consumption['cost']
            
            # Passenger compensation
            direct_costs['passenger_compensation'] += passengers * delay_minutes * self.passenger_compensation_rate
            
            # Cargo value loss (for cargo aircraft)
            if self._is_cargo_aircraft(aircraft_type):
                cargo_value = self._estimate_cargo_value(aircraft_type)
                direct_costs['cargo_value_loss'] += cargo_value * (delay_minutes / 60) * 0.1
        
        # Indirect costs
        indirect_costs = {
            'reputation_damage': self.airline_reputation_cost * len(affected_flights),
            'missed_connection_costs': self._calculate_missed_connection_costs(affected_flights, delay_minutes),
            'operational_efficiency_loss': self._calculate_efficiency_loss(bottleneck_type, delay_minutes)
        }
        
        # Total economic impact
        total_direct = sum(direct_costs.values())
        total_indirect = sum(indirect_costs.values())
        total_economic_impact = total_direct + total_indirect
        
        return {
            'direct_costs': direct_costs,
            'indirect_costs': indirect_costs,
            'total_economic_impact_usd': total_economic_impact,
            'cost_per_minute': total_economic_impact / max(delay_minutes, 1),
            'cost_per_affected_passenger': total_economic_impact / max(sum(f.get('estimated_passengers', 0) for f in affected_flights), 1)
        }
    
    def calculate_environmental_impact(self, bottleneck_data: Dict, 
                                     affected_flights: List[Dict]) -> Dict[str, float]:
        """
        Calculate environmental impact of bottlenecks
        
        Args:
            bottleneck_data: Bottleneck analysis data
            affected_flights: List of affected flight data
            
        Returns:
            Dictionary of environmental impact metrics
        """
        delay_minutes = bottleneck_data.get('delay_minutes', 0)
        
        total_fuel_waste = 0
        total_co2_emissions = 0
        total_nox_emissions = 0
        
        for flight in affected_flights:
            fuel_consumption = self._calculate_flight_fuel_consumption(flight, delay_minutes)
            total_fuel_waste += fuel_consumption['gallons']
            
            # CO2 emissions
            co2_emissions = fuel_consumption['gallons'] * self.co2_per_gallon_fuel
            total_co2_emissions += co2_emissions
            
            # NOx emissions (simplified estimate)
            nox_emissions = fuel_consumption['gallons'] * 0.5  # lbs NOx per gallon (simplified)
            total_nox_emissions += nox_emissions
        
        # Environmental cost
        co2_cost = (total_co2_emissions / self.co2_per_ton) * self.co2_cost_per_ton
        
        return {
            'fuel_waste_gallons': total_fuel_waste,
            'co2_emissions_lbs': total_co2_emissions,
            'co2_emissions_tons': total_co2_emissions / self.co2_per_ton,
            'nox_emissions_lbs': total_nox_emissions,
            'environmental_cost_usd': co2_cost,
            'carbon_footprint_per_passenger': total_co2_emissions / max(sum(f.get('estimated_passengers', 0) for f in affected_flights), 1)
        }
    
    def calculate_operational_impact(self, bottleneck_data: Dict, 
                                   airport_config: Dict) -> Dict[str, float]:
        """
        Calculate operational impact metrics
        
        Args:
            bottleneck_data: Bottleneck analysis data
            airport_config: Airport configuration
            
        Returns:
            Dictionary of operational impact metrics
        """
        bottleneck_type = bottleneck_data.get('type', 'unknown')
        severity = bottleneck_data.get('severity', 0.5)
        delay_minutes = bottleneck_data.get('delay_minutes', 0)
        
        # Operational efficiency metrics
        efficiency_metrics = {
            'runway_utilization_loss': 0,
            'gate_utilization_loss': 0,
            'taxiway_congestion_factor': 0,
            'overall_efficiency_score': 0
        }
        
        if bottleneck_type == 'runway_approach_queue':
            efficiency_metrics['runway_utilization_loss'] = severity * 0.3
            efficiency_metrics['overall_efficiency_score'] = 1.0 - (severity * 0.4)
            
        elif bottleneck_type == 'runway_departure_queue':
            efficiency_metrics['runway_utilization_loss'] = severity * 0.25
            efficiency_metrics['overall_efficiency_score'] = 1.0 - (severity * 0.35)
            
        elif bottleneck_type == 'taxiway_intersection':
            efficiency_metrics['taxiway_congestion_factor'] = severity * 0.5
            efficiency_metrics['overall_efficiency_score'] = 1.0 - (severity * 0.2)
            
        elif bottleneck_type == 'gate_availability':
            efficiency_metrics['gate_utilization_loss'] = severity * 0.4
            efficiency_metrics['overall_efficiency_score'] = 1.0 - (severity * 0.3)
        
        # Calculate cascading effects
        cascade_impact = self._calculate_cascade_effects(bottleneck_data, airport_config)
        
        return {
            'efficiency_metrics': efficiency_metrics,
            'cascade_impact': cascade_impact,
            'operational_disruption_level': min(severity * 2.0, 1.0),
            'recovery_time_minutes': self._estimate_recovery_time(bottleneck_type, severity),
            'mitigation_priority': self._calculate_mitigation_priority(bottleneck_data)
        }
    
    def calculate_passenger_impact(self, affected_flights: List[Dict], 
                                 delay_minutes: float) -> Dict[str, float]:
        """
        Calculate passenger-specific impact metrics
        
        Args:
            affected_flights: List of affected flight data
            delay_minutes: Delay duration in minutes
            
        Returns:
            Dictionary of passenger impact metrics
        """
        total_passengers = sum(f.get('estimated_passengers', 0) for f in affected_flights)
        
        if total_passengers == 0:
            return {
                'passengers_affected': 0,
                'missed_connections': 0,
                'customer_satisfaction_impact': 0,
                'compensation_cost': 0
            }
        
        # Missed connections (15% of passengers per hour of delay)
        missed_connections = total_passengers * 0.15 * (delay_minutes / 60)
        
        # Customer satisfaction impact (0-1 scale)
        satisfaction_impact = min(delay_minutes / 60 * 0.2, 1.0)
        
        # Compensation cost
        compensation_cost = total_passengers * delay_minutes * self.passenger_compensation_rate
        
        # Passenger experience metrics
        experience_metrics = {
            'passengers_affected': total_passengers,
            'missed_connections': missed_connections,
            'customer_satisfaction_impact': satisfaction_impact,
            'compensation_cost': compensation_cost,
            'average_delay_per_passenger': delay_minutes,
            'passenger_disruption_score': satisfaction_impact * 100  # 0-100 scale
        }
        
        return experience_metrics
    
    def calculate_mitigation_cost_benefit(self, bottleneck_data: Dict, 
                                        mitigation_options: List[Dict]) -> List[Dict]:
        """
        Calculate cost-benefit analysis for mitigation options
        
        Args:
            bottleneck_data: Bottleneck analysis data
            mitigation_options: List of mitigation options
            
        Returns:
            List of cost-benefit analyses for each mitigation option
        """
        cost_benefit_analyses = []
        
        for mitigation in mitigation_options:
            implementation_cost = mitigation.get('implementation_cost', 0)
            effectiveness = mitigation.get('estimated_effectiveness', 0.5)
            implementation_time = mitigation.get('implementation_time', 0)
            
            # Calculate benefit (avoided costs)
            avoided_costs = self._calculate_avoided_costs(bottleneck_data, effectiveness)
            
            # Calculate net benefit
            net_benefit = avoided_costs - implementation_cost
            
            # Calculate ROI
            roi = (net_benefit / max(implementation_cost, 1)) * 100
            
            # Calculate payback period
            payback_period = implementation_cost / max(avoided_costs, 1) if avoided_costs > 0 else float('inf')
            
            cost_benefit_analyses.append({
                'mitigation_name': mitigation.get('action', 'Unknown'),
                'implementation_cost': implementation_cost,
                'avoided_costs': avoided_costs,
                'net_benefit': net_benefit,
                'roi_percentage': roi,
                'payback_period_hours': payback_period,
                'effectiveness': effectiveness,
                'implementation_time_minutes': implementation_time,
                'recommendation': 'high' if roi > 100 else 'medium' if roi > 50 else 'low'
            })
        
        return cost_benefit_analyses
    
    def _calculate_flight_fuel_consumption(self, flight: Dict, delay_minutes: float) -> Dict[str, float]:
        """Calculate fuel consumption for a specific flight during delay"""
        aircraft_type = flight.get('aircraft_type', 'B737')
        phase = flight.get('phase', 'ground')
        
        # Fuel burn rates (gallons per hour)
        fuel_rates = {
            'B737': {'idle': 150, 'holding': 225},
            'A320': {'idle': 140, 'holding': 210},
            'B777': {'idle': 300, 'holding': 450},
            'A380': {'idle': 500, 'holding': 750},
            'CRJ9': {'idle': 80, 'holding': 120},
            'E175': {'idle': 75, 'holding': 112}
        }
        
        rates = fuel_rates.get(aircraft_type, fuel_rates['B737'])
        
        if phase == 'holding_pattern':
            fuel_burn_rate = rates['holding']
        else:
            fuel_burn_rate = rates['idle']
        
        fuel_gallons = (fuel_burn_rate / 60) * delay_minutes
        fuel_cost = fuel_gallons * self.fuel_price_per_gallon
        
        return {
            'gallons': fuel_gallons,
            'cost': fuel_cost,
            'burn_rate': fuel_burn_rate
        }
    
    def _is_cargo_aircraft(self, aircraft_type: str) -> bool:
        """Check if aircraft is cargo type"""
        cargo_types = ['B767F', 'B747F', 'B777F', 'A330F']
        return aircraft_type in cargo_types
    
    def _estimate_cargo_value(self, aircraft_type: str) -> float:
        """Estimate cargo value for cargo aircraft"""
        cargo_values = {
            'B767F': 50000,
            'B747F': 100000,
            'B777F': 75000,
            'A330F': 60000
        }
        return cargo_values.get(aircraft_type, 50000)
    
    def _calculate_missed_connection_costs(self, affected_flights: List[Dict], delay_minutes: float) -> float:
        """Calculate costs associated with missed connections"""
        missed_connections = sum(f.get('estimated_passengers', 0) for f in affected_flights) * 0.15 * (delay_minutes / 60)
        return missed_connections * 500  # $500 average cost per missed connection
    
    def _calculate_efficiency_loss(self, bottleneck_type: str, delay_minutes: float) -> float:
        """Calculate operational efficiency loss"""
        efficiency_loss_rates = {
            'runway_approach_queue': 1000,  # USD per minute
            'runway_departure_queue': 800,
            'taxiway_intersection': 500,
            'gate_availability': 1200
        }
        return efficiency_loss_rates.get(bottleneck_type, 500) * delay_minutes
    
    def _calculate_cascade_effects(self, bottleneck_data: Dict, airport_config: Dict) -> Dict[str, float]:
        """Calculate cascading effects of bottlenecks"""
        bottleneck_type = bottleneck_data.get('type', 'unknown')
        severity = bottleneck_data.get('severity', 0.5)
        
        cascade_effects = {
            'downstream_delays': 0,
            'gate_blocking': 0,
            'runway_congestion': 0,
            'taxiway_backup': 0
        }
        
        if bottleneck_type == 'runway_approach_queue':
            cascade_effects['gate_blocking'] = severity * 0.8
            cascade_effects['downstream_delays'] = severity * 0.6
            
        elif bottleneck_type == 'gate_availability':
            cascade_effects['runway_congestion'] = severity * 0.7
            cascade_effects['taxiway_backup'] = severity * 0.5
        
        return cascade_effects
    
    def _estimate_recovery_time(self, bottleneck_type: str, severity: float) -> float:
        """Estimate recovery time for different bottleneck types"""
        base_recovery_times = {
            'runway_approach_queue': 30,
            'runway_departure_queue': 20,
            'taxiway_intersection': 15,
            'gate_availability': 45
        }
        
        base_time = base_recovery_times.get(bottleneck_type, 25)
        return base_time * (1 + severity)
    
    def _calculate_mitigation_priority(self, bottleneck_data: Dict) -> str:
        """Calculate mitigation priority based on bottleneck characteristics"""
        severity = bottleneck_data.get('severity', 0.5)
        delay_minutes = bottleneck_data.get('delay_minutes', 0)
        
        if severity > 0.8 or delay_minutes > 30:
            return 'critical'
        elif severity > 0.6 or delay_minutes > 15:
            return 'high'
        elif severity > 0.4 or delay_minutes > 5:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_avoided_costs(self, bottleneck_data: Dict, effectiveness: float) -> float:
        """Calculate costs that would be avoided by implementing mitigation"""
        delay_minutes = bottleneck_data.get('delay_minutes', 0)
        severity = bottleneck_data.get('severity', 0.5)
        
        # Base avoided costs (simplified calculation)
        base_avoided_costs = delay_minutes * 1000 * severity  # $1000 per minute per severity unit
        
        return base_avoided_costs * effectiveness
