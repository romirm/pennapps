const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const axios = require('axios');
const cron = require('node-cron');
const { generateMockFlights, generateMockConflicts, generateMockTranscripts } = require('./demoData');
require('dotenv').config();

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// OpenSky Network API configuration
const OPENSKY_API_BASE = 'https://opensky-network.org/api';
const OPENSKY_STATES_ENDPOINT = `${OPENSKY_API_BASE}/states/all`;

// Cerebras API configuration
const CEREBRAS_API_KEY = process.env.CEREBRAS_API_KEY;
const CEREBRAS_API_URL = process.env.CEREBRAS_API_URL || 'https://api.cerebras.ai/v1';

// Store current flight data
let currentFlights = new Map();
let flightHistory = [];
let conflicts = [];

// Flight categorization
let enRouteFlights = new Map();
let taxiFlights = new Map();
let incomingFlights = new Map();

// ATC communications by phase
let atcCommunications = {
  enRoute: [],
  taxi: [],
  incoming: []
};

// Calculate distance between two points (Haversine formula)
function calculateDistance(lat1, lon1, lat2, lon2) {
  const R = 6371; // Earth's radius in kilometers
  const dLat = (lat2 - lat1) * Math.PI / 180;
  const dLon = (lon2 - lon1) * Math.PI / 180;
  const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
    Math.cos(lat1 * Math.PI / 180) * Math.cos(lat2 * Math.PI / 180) *
    Math.sin(dLon/2) * Math.sin(dLon/2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
  return R * c; // Distance in kilometers
}

// Calculate altitude difference in feet
function calculateAltitudeDifference(alt1, alt2) {
  return Math.abs(alt1 - alt2);
}

// Categorize flights based on altitude and position
function categorizeFlight(flight) {
  const ordLat = 41.9786;
  const ordLon = -87.9048;
  const distanceFromORD = calculateDistance(flight.latitude, flight.longitude, ordLat, ordLon);
  
  // Taxi flights: on ground and close to airport
  if (flight.onGround && distanceFromORD < 10) {
    return 'taxi';
  }
  // Incoming flights: in air, close to airport, descending
  else if (!flight.onGround && distanceFromORD < 50 && flight.verticalRate < -500) {
    return 'incoming';
  }
  // En route flights: in air, higher altitude, or far from airport
  else if (!flight.onGround && (flight.altitude > 5000 || distanceFromORD > 50)) {
    return 'enRoute';
  }
  // Default to en route for other cases
  else {
    return 'enRoute';
  }
}

// Generate realistic ATC communications based on flight phase
function generateATCCommunications(flights) {
  const communications = {
    enRoute: [],
    taxi: [],
    incoming: []
  };
  
  // Generate communications for each flight category
  Object.keys(flights).forEach(category => {
    const flightList = Array.from(flights[category].values());
    
    flightList.forEach(flight => {
      const timeAgo = Math.floor(Math.random() * 300); // 0-5 minutes ago
      const timestamp = new Date(Date.now() - timeAgo * 1000);
      
      let communication;
      
      switch(category) {
        case 'enRoute':
          communication = generateEnRouteCommunication(flight, timestamp);
          break;
        case 'taxi':
          communication = generateTaxiCommunication(flight, timestamp);
          break;
        case 'incoming':
          communication = generateIncomingCommunication(flight, timestamp);
          break;
      }
      
      if (communication) {
        communications[category].push(communication);
      }
    });
  });
  
  // Sort by timestamp (most recent first)
  Object.keys(communications).forEach(category => {
    communications[category].sort((a, b) => b.timestamp - a.timestamp);
  });
  
  return communications;
}

function generateEnRouteCommunication(flight, timestamp) {
  const communications = [
    {
      id: `enroute-${flight.icao24}-${Date.now()}`,
      timestamp: timestamp,
      type: 'atc',
      speaker: 'ORD Center',
      message: `${flight.callsign}, contact Chicago Center on 135.9`,
      callsign: flight.callsign,
      phase: 'enRoute'
    },
    {
      id: `enroute-${flight.icao24}-${Date.now()}-1`,
      timestamp: new Date(timestamp.getTime() - 30000),
      type: 'pilot',
      speaker: 'Pilot',
      message: `Contact Chicago Center 135.9, ${flight.callsign}`,
      callsign: flight.callsign,
      phase: 'enRoute'
    }
  ];
  
  return communications[Math.floor(Math.random() * communications.length)];
}

function generateTaxiCommunication(flight, timestamp) {
  const communications = [
    {
      id: `taxi-${flight.icao24}-${Date.now()}`,
      timestamp: timestamp,
      type: 'atc',
      speaker: 'ORD Ground',
      message: `${flight.callsign}, taxi to runway 10L via Alpha, hold short of runway 22R`,
      callsign: flight.callsign,
      phase: 'taxi'
    },
    {
      id: `taxi-${flight.icao24}-${Date.now()}-1`,
      timestamp: new Date(timestamp.getTime() - 15000),
      type: 'pilot',
      speaker: 'Pilot',
      message: `Taxi to 10L via Alpha, hold short of 22R, ${flight.callsign}`,
      callsign: flight.callsign,
      phase: 'taxi'
    },
    {
      id: `taxi-${flight.icao24}-${Date.now()}-2`,
      timestamp: new Date(timestamp.getTime() - 45000),
      type: 'atc',
      speaker: 'ORD Ground',
      message: `${flight.callsign}, pushback approved, contact ground 121.9`,
      callsign: flight.callsign,
      phase: 'taxi'
    }
  ];
  
  return communications[Math.floor(Math.random() * communications.length)];
}

function generateIncomingCommunication(flight, timestamp) {
  const communications = [
    {
      id: `incoming-${flight.icao24}-${Date.now()}`,
      timestamp: timestamp,
      type: 'atc',
      speaker: 'ORD Approach',
      message: `${flight.callsign}, descend and maintain 3000, expect ILS approach runway 10L`,
      callsign: flight.callsign,
      phase: 'incoming'
    },
    {
      id: `incoming-${flight.icao24}-${Date.now()}-1`,
      timestamp: new Date(timestamp.getTime() - 20000),
      type: 'pilot',
      speaker: 'Pilot',
      message: `Descend and maintain 3000, expect ILS 10L, ${flight.callsign}`,
      callsign: flight.callsign,
      phase: 'incoming'
    },
    {
      id: `incoming-${flight.icao24}-${Date.now()}-2`,
      timestamp: new Date(timestamp.getTime() - 60000),
      type: 'atc',
      speaker: 'ORD Approach',
      message: `${flight.callsign}, contact approach 119.1`,
      callsign: flight.callsign,
      phase: 'incoming'
    }
  ];
  
  return communications[Math.floor(Math.random() * communications.length)];
}

// Detect potential conflicts
function detectConflicts(flights) {
  const newConflicts = [];
  const flightArray = Array.from(flights.values());
  
  for (let i = 0; i < flightArray.length; i++) {
    for (let j = i + 1; j < flightArray.length; j++) {
      const flight1 = flightArray[i];
      const flight2 = flightArray[j];
      
      if (!flight1.latitude || !flight1.longitude || !flight2.latitude || !flight2.longitude) {
        continue;
      }
      
      const distance = calculateDistance(
        flight1.latitude, flight1.longitude,
        flight2.latitude, flight2.longitude
      );
      
      const altitudeDiff = calculateAltitudeDifference(flight1.altitude || 0, flight2.altitude || 0);
      
      // Conflict criteria: < 3NM (5.56km) horizontal and < 1000ft (305m) vertical
      if (distance < 5.56 && altitudeDiff < 305) {
        const conflict = {
          id: `${flight1.callsign}-${flight2.callsign}-${Date.now()}`,
          flight1: flight1,
          flight2: flight2,
          distance: distance,
          altitudeDifference: altitudeDiff,
          riskLevel: distance < 2 ? 'High' : distance < 4 ? 'Medium' : 'Low',
          timestamp: new Date().toISOString()
        };
        newConflicts.push(conflict);
      }
    }
  }
  
  return newConflicts;
}

// Generate AI recommendation for conflict
async function generateAIRecommendation(conflict) {
  if (!CEREBRAS_API_KEY) {
    // Mock recommendation if no API key
    return {
      summary: `${conflict.flight1.callsign} and ${conflict.flight2.callsign} are in potential conflict`,
      recommendation: 'Suggest altitude change or vector change',
      details: `Aircraft are ${conflict.distance.toFixed(2)}km apart with ${conflict.altitudeDifference.toFixed(0)}ft altitude difference`,
      suggestedAction: 'Request altitude change for one aircraft'
    };
  }

  try {
    const prompt = `As an ATC AI assistant, analyze this potential conflict:
    
    Flight 1: ${conflict.flight1.callsign} at ${conflict.flight1.altitude}ft, heading ${conflict.flight1.heading}°
    Flight 2: ${conflict.flight2.callsign} at ${conflict.flight2.altitude}ft, heading ${conflict.flight2.heading}°
    
    Distance: ${conflict.distance.toFixed(2)}km
    Altitude difference: ${conflict.altitudeDifference.toFixed(0)}ft
    Risk level: ${conflict.riskLevel}
    
    Provide a brief summary, recommendation, and suggested action.`;

    const response = await axios.post(`${CEREBRAS_API_URL}/completions`, {
      model: 'cerebras-llama-2-7b-chat',
      prompt: prompt,
      max_tokens: 200,
      temperature: 0.7
    }, {
      headers: {
        'Authorization': `Bearer ${CEREBRAS_API_KEY}`,
        'Content-Type': 'application/json'
      }
    });

    return {
      summary: response.data.choices[0].text.split('\n')[0],
      recommendation: response.data.choices[0].text.split('\n')[1] || 'Monitor situation',
      details: response.data.choices[0].text.split('\n')[2] || 'Continue monitoring',
      suggestedAction: response.data.choices[0].text.split('\n')[3] || 'No immediate action required'
    };
  } catch (error) {
    console.error('Error calling Cerebras API:', error.message);
    return {
      summary: `${conflict.flight1.callsign} and ${conflict.flight2.callsign} conflict detected`,
      recommendation: 'Monitor and prepare for vector change',
      details: `Distance: ${conflict.distance.toFixed(2)}km, Alt diff: ${conflict.altitudeDifference.toFixed(0)}ft`,
      suggestedAction: 'Request altitude change'
    };
  }
}

// Fetch flight data from OpenSky or use demo data
async function fetchFlightData() {
  try {
    // Try to fetch real data from OpenSky
    const response = await axios.get(OPENSKY_STATES_ENDPOINT, { timeout: 5000 });
    const states = response.data.states;
    
    if (!states || states.length === 0) {
      throw new Error('No data from OpenSky');
    }
    
    const newFlights = new Map();
    
    states.forEach(state => {
      if (state[1] && state[2] && state[6] && state[7]) { // Check for valid data
        const flight = {
          callsign: state[1].trim(),
          icao24: state[0],
          latitude: state[6],
          longitude: state[7],
          altitude: state[13] || 0,
          heading: state[10] || 0,
          velocity: state[9] || 0,
          verticalRate: state[11] || 0,
          timestamp: new Date().toISOString(),
          onGround: state[8] || false
        };
        
        newFlights.set(flight.icao24, flight);
      }
    });
    
    currentFlights = newFlights;
    console.log(`Fetched ${newFlights.size} real flights from OpenSky`);
    
  } catch (error) {
    console.log('OpenSky unavailable, using demo data:', error.message);
    
    // Generate demo data
    const mockFlights = generateMockFlights();
    const newFlights = new Map();
    
    mockFlights.forEach(flight => {
      newFlights.set(flight.icao24, flight);
    });
    
    currentFlights = newFlights;
    console.log(`Generated ${newFlights.size} demo flights`);
  }
  
  // Categorize flights
  enRouteFlights.clear();
  taxiFlights.clear();
  incomingFlights.clear();
  
  currentFlights.forEach(flight => {
    const category = categorizeFlight(flight);
    flight.category = category; // Add category to flight object
    
    switch(category) {
      case 'enRoute':
        enRouteFlights.set(flight.icao24, flight);
        break;
      case 'taxi':
        taxiFlights.set(flight.icao24, flight);
        break;
      case 'incoming':
        incomingFlights.set(flight.icao24, flight);
        break;
    }
  });
  
  // Generate ATC communications
  const categorizedFlights = {
    enRoute: enRouteFlights,
    taxi: taxiFlights,
    incoming: incomingFlights
  };
  
  atcCommunications = generateATCCommunications(categorizedFlights);
  
  console.log(`Categorized flights: ${enRouteFlights.size} en route, ${taxiFlights.size} taxi, ${incomingFlights.size} incoming`);
  
  // Detect conflicts
  const newConflicts = detectConflicts(currentFlights);
  if (newConflicts.length > 0) {
    conflicts = [...conflicts, ...newConflicts];
    
    // Generate AI recommendations for new conflicts
    for (const conflict of newConflicts) {
      conflict.aiRecommendation = await generateAIRecommendation(conflict);
    }
  }
  
  // Store in history (keep last 1000 entries)
  flightHistory.push({
    timestamp: new Date().toISOString(),
    flights: Array.from(currentFlights.values()),
    conflicts: newConflicts
  });
  
  if (flightHistory.length > 1000) {
    flightHistory = flightHistory.slice(-1000);
  }
  
  // Emit to connected clients
  io.emit('flightData', {
    flights: Array.from(currentFlights.values()),
    categorizedFlights: {
      enRoute: Array.from(enRouteFlights.values()),
      taxi: Array.from(taxiFlights.values()),
      incoming: Array.from(incomingFlights.values())
    },
    atcCommunications: atcCommunications,
    conflicts: newConflicts,
    timestamp: new Date().toISOString()
  });
}

// API Routes
app.get('/api/flights', (req, res) => {
  res.json({
    flights: Array.from(currentFlights.values()),
    conflicts: conflicts,
    timestamp: new Date().toISOString()
  });
});

app.get('/api/history', (req, res) => {
  res.json(flightHistory);
});

app.get('/api/transcripts', (req, res) => {
  res.json(generateMockTranscripts());
});

app.get('/api/categorized-flights', (req, res) => {
  res.json({
    enRoute: Array.from(enRouteFlights.values()),
    taxi: Array.from(taxiFlights.values()),
    incoming: Array.from(incomingFlights.values()),
    timestamp: new Date().toISOString()
  });
});

app.get('/api/atc-communications', (req, res) => {
  res.json(atcCommunications);
});

app.post('/api/conflict/resolve', (req, res) => {
  const { conflictId, action } = req.body;
  
  // Remove resolved conflict
  conflicts = conflicts.filter(c => c.id !== conflictId);
  
  io.emit('conflictResolved', { conflictId, action });
  res.json({ success: true });
});

// WebSocket connection handling
io.on('connection', (socket) => {
  console.log('Client connected:', socket.id);
  
  // Send current data to newly connected client
  socket.emit('flightData', {
    flights: Array.from(currentFlights.values()),
    conflicts: conflicts,
    timestamp: new Date().toISOString()
  });
  
  socket.on('disconnect', () => {
    console.log('Client disconnected:', socket.id);
  });
});

// Start fetching flight data every 5 seconds
cron.schedule('*/5 * * * * *', fetchFlightData);

// Initial data fetch
fetchFlightData();

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('OpenSky integration active');
  console.log('WebSocket server ready');
});
