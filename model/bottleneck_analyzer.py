"""
Bottleneck Detection and Analysis System
"""

import math
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from sklearn.cluster import DBSCAN
from flight_processor import FlightPosition, Bottleneck


class BottleneckAnalyzer:
    """Analyzes flight positions to detect bottlenecks"""
    
    def __init__(self):
        self.bottlenecks: List[Bottleneck] = []
    
    def calculate_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """Calculate distance between two points in meters"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lng1_rad = math.radians(lng1)
        lat2_rad = math.radians(lat2)
        lng2_rad = math.radians(lng2)
        
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlng/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def classify_bottleneck_type(self, positions: List[FlightPosition]) -> str:
        """Classify bottleneck type based on position characteristics"""
        if not positions:
            return "unknown"
        
        # Calculate average distance from airport
        avg_distance = 0
        airport_counts = {}
        
        for pos in positions:
            airport_counts[pos.airport] = airport_counts.get(pos.airport, 0) + 1
        
        # Find most common airport
        most_common_airport = max(airport_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate average altitude
        avg_altitude = sum(pos.altitude for pos in positions) / len(positions)
        
        # Calculate average speed (approximate from position changes)
        speeds = []
        for i in range(1, len(positions)):
            dist = self.calculate_distance(
                positions[i-1].lat, positions[i-1].lng,
                positions[i].lat, positions[i].lng
            )
            time_diff = (positions[i].timestamp - positions[i-1].timestamp).total_seconds()
            if time_diff > 0:
                speed = dist / time_diff  # m/s
                speeds.append(speed)
        
        avg_speed = sum(speeds) / len(speeds) if speeds else 0
        
        # Classify based on characteristics
        if avg_altitude < 100 and avg_speed < 20:  # Low altitude, slow speed
            return "taxiway"
        elif avg_altitude < 500 and avg_speed < 50:  # Very low altitude, slow
            return "runway"
        elif avg_altitude < 2000 and avg_speed < 100:  # Low altitude, moderate speed
            return "approach"
        elif avg_altitude < 5000:  # Medium altitude
            return "departure"
        else:
            return "enroute"
    
    def calculate_severity(self, positions: List[FlightPosition]) -> int:
        """Calculate bottleneck severity (1-5) based on aircraft count and duration"""
        aircraft_count = len(positions)
        
        if aircraft_count >= 15:
            return 5
        elif aircraft_count >= 12:
            return 4
        elif aircraft_count >= 9:
            return 3
        elif aircraft_count >= 6:
            return 2
        else:
            return 1
    
    def calculate_duration(self, positions: List[FlightPosition]) -> float:
        """Calculate bottleneck duration in minutes"""
        if len(positions) < 2:
            return 0.0
        
        timestamps = [pos.timestamp for pos in positions]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        duration_seconds = (max_time - min_time).total_seconds()
        return duration_seconds / 60.0  # Convert to minutes
    
    def calculate_confidence(self, positions: List[FlightPosition]) -> float:
        """Calculate confidence score (0.0-1.0) for the bottleneck"""
        aircraft_count = len(positions)
        
        # Base confidence on aircraft count
        if aircraft_count >= 10:
            base_confidence = 0.95
        elif aircraft_count >= 8:
            base_confidence = 0.85
        elif aircraft_count >= 6:
            base_confidence = 0.75
        elif aircraft_count >= 5:
            base_confidence = 0.65
        else:
            base_confidence = 0.5
        
        # Adjust for time consistency
        if len(positions) > 1:
            timestamps = [pos.timestamp for pos in positions]
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            if time_span > 1800:  # More than 30 minutes
                base_confidence *= 0.9
            elif time_span > 3600:  # More than 1 hour
                base_confidence *= 0.8
        
        return min(1.0, base_confidence)
    
    def find_bottlenecks(self, flight_positions: List[FlightPosition], 
                        min_aircraft: int = 2, 
                        max_distance_km: float = 5.0) -> List[Bottleneck]:
        """Find clusters of aircraft that indicate bottlenecks"""
        
        if len(flight_positions) < min_aircraft:
            return []
        
        # Convert to numpy array for clustering
        coords = np.array([[f.lat, f.lng] for f in flight_positions])
        
        # Convert km to degrees (approximate: 1 degree ≈ 111 km)
        eps_degrees = max_distance_km / 111.0
        
        # DBSCAN clustering
        clustering = DBSCAN(eps=eps_degrees, min_samples=min_aircraft).fit(coords)
        
        bottlenecks = []
        cluster_id = 0
        
        # Process each cluster
        for unique_label in set(clustering.labels_):
            if unique_label == -1:  # Skip noise points
                continue
            
            # Get flights in this cluster
            cluster_mask = clustering.labels_ == unique_label
            cluster_flights = [f for i, f in enumerate(flight_positions) if cluster_mask[i]]
            
            if len(cluster_flights) >= min_aircraft:
                # Calculate cluster center
                cluster_coords = coords[cluster_mask]
                center_lat = np.mean(cluster_coords[:, 0])
                center_lng = np.mean(cluster_coords[:, 1])
                
                # Calculate cluster timestamp (average)
                cluster_timestamps = [f.timestamp for f in cluster_flights]
                center_timestamp = min(cluster_timestamps)  # Use earliest time
                
                # Create bottleneck
                bottleneck = Bottleneck(
                    id=cluster_id + 1,
                    lat=float(center_lat),
                    lng=float(center_lng),
                    timestamp=center_timestamp,
                    bottleneck_type=self.classify_bottleneck_type(cluster_flights),
                    severity=self.calculate_severity(cluster_flights),
                    duration_minutes=self.calculate_duration(cluster_flights),
                    confidence=self.calculate_confidence(cluster_flights),
                    aircraft_count=len(cluster_flights),
                    affected_aircraft=cluster_flights
                )
                
                bottlenecks.append(bottleneck)
                cluster_id += 1
        
        # Sort by severity (highest first)
        bottlenecks.sort(key=lambda x: (x.severity, x.aircraft_count), reverse=True)
        
        # Reassign IDs
        for i, bottleneck in enumerate(bottlenecks):
            bottleneck.id = i + 1
        
        self.bottlenecks = bottlenecks
        return bottlenecks
    
    def analyze_flights(self, flights: List[FlightPosition]) -> Dict:
        """Main analysis method"""
        print(f"Analyzing {len(flights)} flight positions...")
        
        # Group flights by time windows (10-minute intervals)
        time_windows = self._group_by_time_windows(flights, window_minutes=10)
        
        all_bottlenecks = []
        
        for window_start, window_flights in time_windows.items():
            if len(window_flights) >= 2:  # Only analyze windows with enough flights
                window_bottlenecks = self.find_bottlenecks(window_flights)
                all_bottlenecks.extend(window_bottlenecks)
        
        # Remove duplicates and merge nearby bottlenecks
        merged_bottlenecks = self._merge_nearby_bottlenecks(all_bottlenecks, merge_distance_km=2.0)
        
        self.bottlenecks = merged_bottlenecks
        
        # Calculate summary statistics
        total_flight_points = len(flights)
        predicted_paths = len(set(f.callsign for f in flights))
        bottlenecks_detected = len(merged_bottlenecks)
        high_severity = len([b for b in merged_bottlenecks if b.severity >= 4])
        
        return {
            'total_flight_points': total_flight_points,
            'predicted_paths': predicted_paths,
            'bottlenecks_detected': bottlenecks_detected,
            'high_severity_bottlenecks': high_severity,
            'bottlenecks': merged_bottlenecks
        }
    
    def _group_by_time_windows(self, flights: List[FlightPosition], window_minutes: int = 10) -> Dict:
        """Group flights by time windows"""
        if not flights:
            return {}
        
        # Find time range
        timestamps = [f.timestamp for f in flights]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        # Create time windows
        windows = {}
        current_time = min_time
        
        while current_time < max_time:
            window_end = current_time + timedelta(minutes=window_minutes)
            
            # Find flights in this window
            window_flights = [
                f for f in flights 
                if current_time <= f.timestamp < window_end
            ]
            
            if window_flights:
                windows[current_time] = window_flights
            
            current_time = window_end
        
        return windows
    
    def _merge_nearby_bottlenecks(self, bottlenecks: List[Bottleneck], 
                                 merge_distance_km: float = 5.0) -> List[Bottleneck]:
        """Merge bottlenecks that are close to each other"""
        if len(bottlenecks) <= 1:
            return bottlenecks
        
        merged = []
        used = set()
        
        for i, bottleneck in enumerate(bottlenecks):
            if i in used:
                continue
            
            # Find nearby bottlenecks to merge
            nearby_indices = []
            for j, other in enumerate(bottlenecks):
                if j != i and j not in used:
                    distance = self.calculate_distance(
                        bottleneck.lat, bottleneck.lng,
                        other.lat, other.lng
                    )
                    if distance <= merge_distance_km * 1000:  # Convert to meters
                        nearby_indices.append(j)
            
            if nearby_indices:
                # Merge with nearby bottlenecks
                all_aircraft = bottleneck.affected_aircraft.copy()
                for j in nearby_indices:
                    all_aircraft.extend(bottlenecks[j].affected_aircraft)
                    used.add(j)
                
                # Create merged bottleneck
                merged_bottleneck = Bottleneck(
                    id=len(merged) + 1,
                    lat=bottleneck.lat,
                    lng=bottleneck.lng,
                    timestamp=bottleneck.timestamp,
                    bottleneck_type=bottleneck.bottleneck_type,
                    severity=max(bottleneck.severity, max(bottlenecks[j].severity for j in nearby_indices)),
                    duration_minutes=max(bottleneck.duration_minutes, max(bottlenecks[j].duration_minutes for j in nearby_indices)),
                    confidence=min(bottleneck.confidence, min(bottlenecks[j].confidence for j in nearby_indices)),
                    aircraft_count=len(all_aircraft),
                    affected_aircraft=all_aircraft
                )
                
                merged.append(merged_bottleneck)
                used.add(i)
            else:
                merged.append(bottleneck)
                used.add(i)
        
        return merged
