import asyncio
import time
from typing import Dict, List, Optional
import aiohttp
import json
from datetime import datetime, timedelta
import math

import requests
from typing import Any, Union
from dataclasses import asdict

# Optional imports - these may not be available in all environments
# Exa functionality disabled by user preference
Exa = None

try:
    from cerebras.cloud.sdk import Cerebras
except ImportError:
    Cerebras = None
    print(
        "Warning: cerebras-cloud-sdk not available. Cerebras functionality will be disabled."
    )


class ADSBExchangeGroundClient:
    def __init__(self):
        self.base_url = "https://api.adsb.lol"
        self.session = None

        # Rate limiting - conservative for free API
        self.requests_made = 0
        self.last_reset = datetime.now().replace(hour=0, minute=0, second=0)
        self.min_interval = 1  # 1 second between requests for adsb.lol
        self.last_request_time = 0

        # Caching
        self.cache = {}
        self.cache_duration = 20  # 20 second cache

        # Airport registry
        self.known_airports = {}

    async def __aenter__(self):
        # Simple headers for adsb.lol API
        headers = {
            "User-Agent": "Airport-Ground-Tracker/1.0",
            "Accept": "application/json",
            "Cache-Control": "no-cache",
        }
        self.session = aiohttp.ClientSession(headers=headers)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def register_airport(self, icao: str, centroid: Dict[str, float]):
        """Register an airport's centroid coordinates"""
        self.known_airports[icao] = centroid
        print(
            f"Registered {icao} with adsb.lol monitoring at lat={centroid['lat']}, lon={centroid['lon']}"
        )

    async def _rate_limit_check(self):
        """Minimal rate limiting - ADSBX is more permissive"""
        now = datetime.now()

        if now.date() > self.last_reset.date():
            self.requests_made = 0
            self.last_reset = now

        # Much more permissive rate limiting
        time_since_last = time.time() - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)

        return True

    def _get_cache_key(self, airport_icao: str, centroid: Dict[str, float]) -> str:
        """Generate cache key for specific airport and centroid"""
        return f"adsbx_{airport_icao}:{centroid['lat']},{centroid['lon']}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False

        cache_time = self.cache[cache_key]["timestamp"]
        age = time.time() - cache_time
        return age < self.cache_duration

    def _calculate_distance_nm(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """Calculate distance between two points in nautical miles using Haversine formula"""
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.asin(math.sqrt(a))

        # Earth's radius in nautical miles
        earth_radius_nm = 3440.065  # 1 nautical mile = 1.852 km, Earth radius = 6371 km
        distance_nm = earth_radius_nm * c

        return distance_nm

    async def fetch_airport_data(
        self, airport_icao: str, centroid: Dict[str, float] = None
    ) -> List[Dict]:
        """Fetch all aircraft data from ADS-B Exchange using centroid and radius"""

        if centroid is None:
            centroid = self.known_airports.get(airport_icao)
            if centroid is None:
                raise ValueError(f"Airport {airport_icao} not registered")

        cache_key = self._get_cache_key(airport_icao, centroid)

        # Check cache first
        if self._is_cache_valid(cache_key):
            cached_data = self.cache[cache_key]["data"]
            cache_age = time.time() - self.cache[cache_key]["timestamp"]
            print(
                f"Using cached ADS-B data for {airport_icao} (age: {cache_age:.1f}s, {len(cached_data)} aircraft)"
            )
            return cached_data

        # Rate limit check
        await self._rate_limit_check()

        try:
            # Use centroid coordinates directly
            center_lat = centroid["lat"]
            center_lon = centroid["lon"]

            # adsb.lol API endpoint format: /v2/point/{lat}/{lon}/{radius}
            # Convert nautical miles to kilometers (1 nm = 1.852 km)
            # API requires integer radius, so round to nearest km
            radius_km = int(
                round(5 * 1.852)
            )  # 5 nautical miles in km, rounded to integer
            url = f"{self.base_url}/v2/point/{center_lat}/{center_lon}/{radius_km}"
            params = {}

            print(
                f"Making adsb.lol request for {airport_icao} at lat={center_lat}, lon={center_lon}..."
            )
            self.last_request_time = time.time()

            async with self.session.get(url, params=params) as response:
                self.requests_made += 1

                if response.status == 200:
                    data = await response.json()

                    # adsb.lol returns aircraft in 'ac' array - return exactly what the API gives us
                    aircraft_list = data.get("ac", [])
                    #  print(aircraft_list[0])

                    # DEBUG: Show aircraft count and basic info
                    print(f"\n🔍 DEBUG: API returned {len(aircraft_list)} aircraft")
                    if len(aircraft_list) > 0:
                        print(
                            f"Sample aircraft: {aircraft_list[0].get('flight', 'N/A')} at {aircraft_list[0].get('lat', 'N/A')}, {aircraft_list[0].get('lon', 'N/A')}"
                        )
                        print(f"Request URL: {url}")

                    # Cache the result
                    self.cache[cache_key] = {
                        "data": aircraft_list,
                        "timestamp": time.time(),
                        "airport": airport_icao,
                    }

                    print(f"✅ adsb.lol: {len(aircraft_list)} aircraft returned")
                    return aircraft_list

                elif response.status == 429:
                    print(f"❌ Rate limited by adsb.lol")
                    await asyncio.sleep(30)
                    return []

                elif response.status == 403:
                    print(f"❌ Forbidden - adsb.lol access denied")
                    return []

                else:
                    print(f"❌ adsb.lol error: HTTP {response.status}")
                    return []

        except Exception as e:
            print(f"❌ Error fetching adsb.lol data for {airport_icao}: {e}")
            return []

    def get_aircraft_summary(self, aircraft: List[Dict]) -> Dict:
        """Summary of raw aircraft data from ADS-B Exchange"""
        if not aircraft:
            return {
                "total_aircraft": 0,
                "aircraft_types": {},
                "callsigns": [],
                "speed_distribution": {},
                "altitude_distribution": {},
            }

        speeds = []
        altitudes = []

        for a in aircraft:
            gs = a.get("gs")
            if gs is not None:
                try:
                    speeds.append(float(gs))
                except (ValueError, TypeError):
                    pass

            alt = a.get("alt_baro")
            if alt is not None:
                try:
                    altitudes.append(float(alt))
                except (ValueError, TypeError):
                    pass
        aircraft_types = [a.get("t") for a in aircraft if a.get("t")]
        callsigns = [a.get("flight") for a in aircraft if a.get("flight")]

        # Aircraft type distribution
        type_dist = {}
        for ac_type in aircraft_types:
            type_dist[ac_type] = type_dist.get(ac_type, 0) + 1

        # Speed distribution
        speed_ranges = {
            "stationary (0-2 kts)": len([s for s in speeds if 0 <= s <= 2]),
            "slow (3-50 kts)": len([s for s in speeds if 3 <= s <= 50]),
            "medium (51-200 kts)": len([s for s in speeds if 51 <= s <= 200]),
            "fast (200+ kts)": len([s for s in speeds if s > 200]),
        }

        # Altitude distribution
        altitude_ranges = {
            "ground (0-500 ft)": len([a for a in altitudes if 0 <= a <= 500]),
            "low (501-5000 ft)": len([a for a in altitudes if 501 <= a <= 5000]),
            "medium (5001-20000 ft)": len([a for a in altitudes if 5001 <= a <= 20000]),
            "high (20000+ ft)": len([a for a in altitudes if a > 20000]),
        }

        return {
            "total_aircraft": len(aircraft),
            "aircraft_types": dict(
                sorted(type_dist.items(), key=lambda x: x[1], reverse=True)[:10]
            ),
            "callsigns": callsigns[:10],
            "speed_distribution": speed_ranges,
            "altitude_distribution": altitude_ranges,
            "avg_speed_kts": sum(speeds) / len(speeds) if speeds else 0,
            "avg_altitude_ft": sum(altitudes) / len(altitudes) if altitudes else 0,
        }


# Comprehensive airport centroids for ADS-B Exchange
AIRPORT_CENTROIDS = {
    "KPHL": {"lat": 39.8720, "lon": -75.2407},  # Philadelphia International Airport
    "KJFK": {"lat": 40.6413, "lon": -73.7781},  # John F. Kennedy International Airport
    "KLGA": {"lat": 40.7769, "lon": -73.8740},  # LaGuardia Airport
    "KEWR": {"lat": 40.6895, "lon": -74.1745},  # Newark Liberty International Airport
    "KLAX": {"lat": 33.9425, "lon": -118.4081},  # Los Angeles International Airport
    "KORD": {"lat": 41.9786, "lon": -87.9048},  # Chicago O'Hare International Airport
}


# Usage example with adsb.lol API
async def adsbx_example():
    async with ADSBExchangeGroundClient() as client:
        # Register airports
        for icao, centroid in AIRPORT_CENTROIDS.items():
            client.register_airport(icao, centroid)

        target_airport = "KPHL"

        print(f"Starting adsb.lol monitoring for {target_airport}")

        while True:
            try:
                aircraft = await client.fetch_airport_data(target_airport)

                summary = client.get_aircraft_summary(aircraft)

                print(f"\n📊 {target_airport} Aircraft Operations (adsb.lol):")
                print(f"   Total aircraft: {summary['total_aircraft']}")
                print(f"   Avg speed: {summary['avg_speed_kts']:.1f} kts")
                print(f"   Avg altitude: {summary['avg_altitude_ft']:.0f} ft")

                if summary["aircraft_types"]:
                    print(
                        f"   Aircraft types: {dict(list(summary['aircraft_types'].items())[:3])}"
                    )

                print(f"   Speed distribution:")
                for speed_range, count in summary["speed_distribution"].items():
                    if count > 0:
                        print(f"     {speed_range}: {count}")

                print(f"   Altitude distribution:")
                for alt_range, count in summary["altitude_distribution"].items():
                    if count > 0:
                        print(f"     {alt_range}: {count}")

                if summary["callsigns"]:
                    print(f"   Sample callsigns: {', '.join(summary['callsigns'][:5])}")

                await asyncio.sleep(20)  # Update every 20 seconds

            except KeyboardInterrupt:
                print("\nStopping adsb.lol monitoring...")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(10)


# Test function with KPHL as the test case
async def run():
    """Test the adsb.lol API implementation with KPHL (Philadelphia International Airport)"""
    # Use KPHL coordinates from our predefined centroids
    test_centroid = AIRPORT_CENTROIDS["KJFK"]

    print("Testing adsb.lol API...")
    print(
        f"Testing coordinates: lat={test_centroid['lat']}, lon={test_centroid['lon']} (KPHL - Philadelphia)"
    )

    try:
        async with ADSBExchangeGroundClient() as client:
            client.register_airport("KJFK", test_centroid)
            aircraft = await client.fetch_airport_data("KPHL", test_centroid)

            if aircraft:
                print(f"✅ Success! Found {len(aircraft)} aircraft")
                summary = client.get_aircraft_summary(aircraft)
                print(f"   Total aircraft: {summary['total_aircraft']}")
                print(f"   Avg speed: {summary['avg_speed_kts']:.1f} kts")
                print(f"   Avg altitude: {summary['avg_altitude_ft']:.0f} ft")

                # Show all aircraft
                for i, ac in enumerate(aircraft):
                    if ac.get("alt_baro") != "ground":
                        print(
                            f"   Aircraft {i+1}: {ac.get('flight', 'N/A')} ({ac.get('t', 'N/A')}) - {ac.get('gs', 'N/A')} kts, {ac.get('alt_baro', 'N/A')} ft, lat: {ac.get('lat', 'N/A')}, lon: {ac.get('lon', 'N/A')}"
                        )
                    else:
                        print(
                            f"   GROUNDED AIRCRAFT {i+1}: {ac.get('flight', 'N/A')} ({ac.get('t', 'N/A')}) - {ac.get('gs', 'N/A')} kts, {ac.get('alt_baro', 'N/A')} ft, lat: {ac.get('lat', 'N/A')}, lon: {ac.get('lon', 'N/A')}"
                        )
                    #    print(f"   GROUNDED AIRCRAFT {i+1}: {ac.get('flight', 'N/A')} ({ac.get('t', 'N/A')}) - {ac.get('gs', 'N/A')} kts, {ac.get('alt_baro', 'N/A')} ft, lat: {ac.get('lat', 'N/A')}, lon: {ac.get('lon', 'N/A')}")

                # Create nested list of flight numbers and aircraft numbers
                flight_data = {"ground": {}, "air": {}}
                for ac in aircraft:
                    if ac.get("alt_baro") != "ground":
                        flight_number = ac.get("flight", "N/A")
                        aircraft_type = ac.get("t", "N/A")
                        lat = ac.get("lat", "N/A")
                        lon = ac.get("lon", "N/A")
                        flight_data["air"][flight_number] = {
                            "aircraft_type": aircraft_type,
                            "lat": lat,
                            "lon": lon,
                        }
                    else:
                        flight_number = ac.get("flight", "N/A")
                        aircraft_type = ac.get("t", "N/A")
                        lat = ac.get("lat", "N/A")
                        lon = ac.get("lon", "N/A")
                        flight_data["ground"][flight_number] = {
                            "aircraft_type": aircraft_type,
                            "lat": lat,
                            "lon": lon,
                        }

                return flight_data
            else:
                print("ℹ️  No aircraft found in the test area (this is normal)")
                return []

    except Exception as e:
        print(f"❌ Test failed: {e}")


class PlaneMonitor:
    def __init__(self):
        self.previous_planes = {"ground": {}, "air": {}}
        self.kjfk_centroid = AIRPORT_CENTROIDS["KJFK"]

    async def fetch_planes(self):
        """Fetch planes at the specified airport and detect entering/leaving aircraft"""
        try:
            async with ADSBExchangeGroundClient() as client:
                client.register_airport("KJFK", self.kjfk_centroid)
                aircraft = await client.fetch_airport_data("KJFK", self.kjfk_centroid)

                if aircraft:
                    # Create current state similar to run() function structure
                    current_planes = {"ground": {}, "air": {}}

                    for ac in aircraft:
                        if ac.get("alt_baro") != "ground":
                            flight_number = ac.get("flight", "N/A")
                            aircraft_type = ac.get("t", "N/A")
                            lat = ac.get("lat", "N/A")
                            lon = ac.get("lon", "N/A")
                            current_planes["air"][flight_number] = {
                                "aircraft_type": aircraft_type,
                                "lat": lat,
                                "lon": lon,
                                "speed": ac.get("gs", "N/A"),
                                "altitude": ac.get("alt_baro", "N/A"),
                                "heading": ac.get("track", "N/A"),
                            }
                        else:
                            flight_number = ac.get("flight", "N/A")
                            aircraft_type = ac.get("t", "N/A")
                            lat = ac.get("lat", "N/A")
                            lon = ac.get("lon", "N/A")
                            current_planes["ground"][flight_number] = {
                                "aircraft_type": aircraft_type,
                                "lat": lat,
                                "lon": lon,
                                "speed": ac.get("gs", "N/A"),
                                "altitude": ac.get("alt_baro", "N/A"),
                                "heading": ac.get("track", "N/A"),
                            }

                    # Detect changes (entering/leaving aircraft)
                    changes = self._detect_changes(current_planes)

                    # Update previous state
                    self.previous_planes = current_planes

                    # Return current state with change information
                    return {
                        "current_planes": current_planes,
                        "changes": changes,
                        "timestamp": datetime.now().isoformat(),
                        "total_aircraft": len(aircraft),
                    }
                else:
                    # No aircraft found
                    changes = self._detect_changes({"ground": {}, "air": {}})
                    self.previous_planes = {"ground": {}, "air": {}}

                    return {
                        "current_planes": {"ground": {}, "air": {}},
                        "changes": changes,
                        "timestamp": datetime.now().isoformat(),
                        "total_aircraft": 0,
                    }

        except Exception as e:
            print(f"❌ Error in fetch_planes: {e}")
            return {
                "current_planes": {"ground": {}, "air": {}},
                "changes": {"entered": [], "left": []},
                "timestamp": datetime.now().isoformat(),
                "total_aircraft": 0,
                "error": str(e),
            }

    def _detect_changes(self, current_planes):
        """Detect aircraft that entered or left the airport area"""
        entered = []
        left = []

        # Check for aircraft that entered
        for category in ["ground", "air"]:
            for flight_number, plane_data in current_planes[category].items():
                if flight_number not in self.previous_planes[category]:
                    entered.append(
                        {
                            "flight_number": flight_number,
                            "category": category,
                            "data": plane_data,
                            "action": "entered",
                        }
                    )

        # Check for aircraft that left
        for category in ["ground", "air"]:
            for flight_number, plane_data in self.previous_planes[category].items():
                if flight_number not in current_planes[category]:
                    left.append(
                        {
                            "flight_number": flight_number,
                            "category": category,
                            "data": plane_data,
                            "action": "left",
                        }
                    )

        return {"entered": entered, "left": left}


# Global monitor instance for the standalone fetch_planes function
_plane_monitor = PlaneMonitor()


async def fetch_planes():
    """
    Standalone function to fetch planes at specified airport.
    Returns a dictionary with the same structure as run() function,
    but includes additional change detection information.
    """
    # Update the global monitor with the new airport if different
    global _plane_monitor
    if _plane_monitor.airport_code != airport_code.upper():
        _plane_monitor = PlaneMonitor(airport_code)
    
    return await _plane_monitor.fetch_planes()


class CerebrasClient:
    def __init__(self):
        # Exa functionality disabled by user preference
        self.exa = None

        if Cerebras is not None:
            # Load API key from environment
            import os

            api_key = os.getenv("CEREBRAS_API_KEY")
            if api_key:
                self.cerebras = Cerebras(api_key=api_key)
            else:
                print("Warning: CEREBRAS_API_KEY not found in environment variables")
                self.cerebras = None
        else:
            self.cerebras = None

    def search_air(self, flightNumbers):
        if self.exa is None:
            # Exa service disabled - return empty flight info
            return {num: "Flight info lookup disabled" for num in flightNumbers}

        flightInfo = {}
        for num in flightNumbers:
            query = f"FlightAware [{num}]"
            result = self.exa.search_and_contents(
                query, type="auto", num_results=1, text={"max_characters": 10000}
            )
            flightInfo[num] = result.results
        return flightInfo

    def ask_ai(self, prompt):
        if self.cerebras is None:
            return "Cerebras AI service not available"

        chat_completion = self.cerebras.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="gpt-oss-120b",  # qwen-3-32b
            max_tokens=600,
            temperature=0.2,
        )
        return chat_completion.choices[0].message.content

    def interpret_air(self, flightNumbers):
        flightInfo = self.search_air(flightNumbers)
        prompt = f"""
        You are an expert in aviation and airport traffic control at John F Kennedy International Airport. You have been given the following flight numbers: {flightNumbers}, as well as 
        top web searches for each of these flights. Synthesize this given information to determine the following about the given list of flights. Do NOT make up any information and only use the information provided.
        
        - Estimated arrival time
        - Expected gate at landing
        - Expected runway at landing
        
        Return your response in a JSON format with the following keys:
        - flight_number: the flight number
        - arrival_time: the estimated arrival time
        - gate: the expected gate at landing
        - runway: the expected runway at landing
        
        If a flight is not scheduled to land at JFK, do not include it in your response.
        """
        response = self.ask_ai(prompt)
        return response


async def monitor_kjfk_example():
    """Example of using fetch_planes function to monitor KJFK every 10 seconds"""
    print("Starting KJFK plane monitoring...")
    print("Press Ctrl+C to stop")

    try:
        while True:
            result = await fetch_planes()

            print(f"\n📊 KJFK Aircraft Status - {result['timestamp']}")
            print(f"   Total aircraft: {result['total_aircraft']}")

            # Show current planes
            if result["current_planes"]["air"]:
                print(f"   Aircraft in air: {len(result['current_planes']['air'])}")
                for flight, data in result["current_planes"]["air"].items():
                    print(
                        f"     ✈️  {flight} ({data['aircraft_type']}) - {data['speed']} kts, {data['altitude']} ft"
                    )

            if result["current_planes"]["ground"]:
                print(
                    f"   Aircraft on ground: {len(result['current_planes']['ground'])}"
                )
                for flight, data in result["current_planes"]["ground"].items():
                    print(
                        f"     🛬 {flight} ({data['aircraft_type']}) - {data['speed']} kts"
                    )

            # Show changes
            if result["changes"]["entered"]:
                print(f"   🟢 Aircraft entered:")
                for change in result["changes"]["entered"]:
                    print(f"     + {change['flight_number']} ({change['category']})")

            if result["changes"]["left"]:
                print(f"   🔴 Aircraft left:")
                for change in result["changes"]["left"]:
                    print(f"     - {change['flight_number']} ({change['category']})")

            # Wait 10 seconds before next check
            await asyncio.sleep(10)

    except KeyboardInterrupt:
        print("\nStopping KJFK monitoring...")
    except Exception as e:
        print(f"Error in monitoring: {e}")


if __name__ == "__main__":
    # Run the test first, then the full example

    res = asyncio.run(run())
    # print(res['air'])
    # print(res['ground'])
    airKeys = list(res["air"].keys())
    airKeys = [key.strip() for key in airKeys]
    cerebras = CerebrasClient()
    response = cerebras.search_air(airKeys)
    for key in response:
        print(key)
        print(response[key])
        print("\n")

    # Uncomment the line below to run the KJFK monitoring example
    # asyncio.run(monitor_kjfk_example())

# print("\n=== Running Full Example ===")
# asyncio.run(adsbx_example())
