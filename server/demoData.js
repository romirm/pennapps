// Demo data generator for enhanced demo experience
const generateMockFlights = () => {
  const baseLat = 41.9786; // O'Hare latitude
  const baseLon = -87.9048; // O'Hare longitude
  
  const airlines = ['AAL', 'UAL', 'DAL', 'SWA', 'JBU', 'F9', 'NK', 'AS'];
  const flightNumbers = ['123', '456', '789', '101', '202', '303', '404', '505'];
  
  const flights = [];
  
  // Generate 8-12 mock flights around O'Hare
  const numFlights = Math.floor(Math.random() * 5) + 8;
  
  for (let i = 0; i < numFlights; i++) {
    const airline = airlines[Math.floor(Math.random() * airlines.length)];
    const flightNum = flightNumbers[Math.floor(Math.random() * flightNumbers.length)];
    
    // Generate position within 50km radius of ORD
    const angle = Math.random() * 2 * Math.PI;
    const distance = Math.random() * 50; // km
    const lat = baseLat + (distance * Math.cos(angle) / 111); // Rough conversion
    const lon = baseLon + (distance * Math.sin(angle) / (111 * Math.cos(baseLat * Math.PI / 180)));
    
    const flight = {
      callsign: `${airline}${flightNum}`,
      icao24: `abc${i.toString().padStart(3, '0')}`,
      latitude: lat,
      longitude: lon,
      altitude: Math.floor(Math.random() * 30000) + 10000, // 10k-40k feet
      heading: Math.floor(Math.random() * 360),
      velocity: Math.floor(Math.random() * 200) + 150, // 150-350 kts
      verticalRate: Math.floor(Math.random() * 2000) - 1000, // -1000 to +1000 fpm
      timestamp: new Date().toISOString(),
      onGround: false
    };
    
    flights.push(flight);
  }
  
  return flights;
};

const generateMockConflicts = (flights) => {
  const conflicts = [];
  
  // Generate 1-2 mock conflicts
  const numConflicts = Math.floor(Math.random() * 2) + 1;
  
  for (let i = 0; i < numConflicts && flights.length >= 2; i++) {
    const flight1 = flights[Math.floor(Math.random() * flights.length)];
    const flight2 = flights[Math.floor(Math.random() * flights.length)];
    
    if (flight1.icao24 !== flight2.icao24) {
      const distance = Math.random() * 3 + 1; // 1-4 km (simulating conflict)
      const altDiff = Math.random() * 500 + 100; // 100-600 ft difference
      
      const conflict = {
        id: `conflict-${Date.now()}-${i}`,
        flight1: flight1,
        flight2: flight2,
        distance: distance,
        altitudeDifference: altDiff,
        riskLevel: distance < 2 ? 'High' : distance < 3 ? 'Medium' : 'Low',
        timestamp: new Date().toISOString(),
        aiRecommendation: {
          summary: `${flight1.callsign} and ${flight2.callsign} are on converging paths`,
          recommendation: distance < 2 ? 'Immediate vector change required' : 'Monitor closely and prepare for vector change',
          details: `Aircraft are ${distance.toFixed(2)}km apart with ${altDiff.toFixed(0)}ft altitude difference`,
          suggestedAction: distance < 2 ? 'Request immediate altitude change for one aircraft' : 'Suggest heading change or altitude adjustment'
        }
      };
      
      conflicts.push(conflict);
    }
  }
  
  return conflicts;
};

const generateMockTranscripts = () => {
  const transcripts = [
    {
      id: 'transcript-1',
      timestamp: new Date(Date.now() - 30000),
      type: 'atc',
      speaker: 'ORD Ground',
      message: 'American 123, taxi to runway 10L via Alpha, hold short of runway 22R',
      callsign: 'AAL123'
    },
    {
      id: 'transcript-2',
      timestamp: new Date(Date.now() - 25000),
      type: 'pilot',
      speaker: 'Pilot',
      message: 'Taxi to 10L via Alpha, hold short of 22R, American 123',
      callsign: 'AAL123'
    },
    {
      id: 'transcript-3',
      timestamp: new Date(Date.now() - 20000),
      type: 'atc',
      speaker: 'ORD Tower',
      message: 'United 456, cleared for takeoff runway 10L, wind 090 at 12',
      callsign: 'UAL456'
    },
    {
      id: 'transcript-4',
      timestamp: new Date(Date.now() - 15000),
      type: 'pilot',
      speaker: 'Pilot',
      message: 'Cleared for takeoff 10L, United 456',
      callsign: 'UAL456'
    },
    {
      id: 'transcript-5',
      timestamp: new Date(Date.now() - 10000),
      type: 'atc',
      speaker: 'ORD Approach',
      message: 'Delta 789, climb and maintain 5000, contact departure 121.9',
      callsign: 'DAL789'
    },
    {
      id: 'transcript-6',
      timestamp: new Date(Date.now() - 5000),
      type: 'pilot',
      speaker: 'Pilot',
      message: 'Climb and maintain 5000, contact departure 121.9, Delta 789',
      callsign: 'DAL789'
    }
  ];
  
  return transcripts;
};

module.exports = {
  generateMockFlights,
  generateMockConflicts,
  generateMockTranscripts
};
