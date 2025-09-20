import React, { useState, useEffect } from 'react';
import io from 'socket.io-client';
import RadarMap from './components/RadarMap';
import TimelinePanel from './components/TimelinePanel';
import RecommendationCards from './components/RecommendationCards';
import ControlPanel from './components/ControlPanel';
import ATCPhaseSelector from './components/ATCPhaseSelector';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [flights, setFlights] = useState([]);
  const [categorizedFlights, setCategorizedFlights] = useState({
    enRoute: [],
    taxi: [],
    incoming: []
  });
  const [conflicts, setConflicts] = useState([]);
  const [selectedFlight, setSelectedFlight] = useState(null);
  const [isLive, setIsLive] = useState(true);
  const [playbackTime, setPlaybackTime] = useState(0);
  const [atcCommunications, setAtcCommunications] = useState({
    enRoute: [],
    taxi: [],
    incoming: []
  });
  const [activePhase, setActivePhase] = useState('all'); // 'all', 'enRoute', 'taxi', 'incoming'

  useEffect(() => {
    // Initialize socket connection
    const newSocket = io(API_BASE_URL);

    newSocket.on('flightData', (data) => {
      setFlights(data.flights);
      setCategorizedFlights(data.categorizedFlights || {
        enRoute: [],
        taxi: [],
        incoming: []
      });
      setAtcCommunications(data.atcCommunications || {
        enRoute: [],
        taxi: [],
        incoming: []
      });
      setConflicts(data.conflicts);
    });

    newSocket.on('conflictResolved', (data) => {
      setConflicts(prev => prev.filter(c => c.id !== data.conflictId));
    });

    // Fetch ATC communications
    fetch(`${API_BASE_URL}/api/atc-communications`)
      .then(res => res.json())
      .then(data => setAtcCommunications(data))
      .catch(err => console.error('Error fetching ATC communications:', err));

    return () => {
      newSocket.close();
    };
  }, []);

  const handleConflictResolution = async (conflictId, action) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/conflict/resolve`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ conflictId, action }),
      });
      
      if (response.ok) {
        setConflicts(prev => prev.filter(c => c.id !== conflictId));
      }
    } catch (error) {
      console.error('Error resolving conflict:', error);
    }
  };

  const handleFlightSelect = (flight) => {
    setSelectedFlight(flight);
  };

  const toggleLiveMode = () => {
    setIsLive(!isLive);
  };

  return (
    <div className="App bg-radar-dark min-h-screen text-radar-green">
      {/* Header */}
      <header className="bg-black bg-opacity-50 p-4 border-b border-radar-green">
        <div className="flex justify-between items-center">
          <h1 className="text-2xl font-bold font-orbitron">
            ATC AI Co-Pilot Dashboard
          </h1>
          <div className="flex items-center space-x-4">
            <div className={`px-3 py-1 rounded ${isLive ? 'bg-green-600' : 'bg-gray-600'}`}>
              {isLive ? 'LIVE' : 'PLAYBACK'}
            </div>
            <button
              onClick={toggleLiveMode}
              className="px-4 py-2 bg-radar-green text-black rounded hover:bg-radar-glow transition-colors"
            >
              {isLive ? 'Switch to Playback' : 'Switch to Live'}
            </button>
          </div>
        </div>
      </header>

      <div className="flex h-screen">
        {/* Main Radar View */}
        <div className="flex-1 relative">
          <RadarMap
            flights={flights}
            categorizedFlights={categorizedFlights}
            conflicts={conflicts}
            selectedFlight={selectedFlight}
            onFlightSelect={handleFlightSelect}
            isLive={isLive}
            activePhase={activePhase}
          />
          
          {/* Flight Info Overlay */}
          {selectedFlight && (
            <div className="absolute top-4 right-4 bg-black bg-opacity-80 p-4 rounded border border-radar-green">
              <h3 className="font-bold text-lg">{selectedFlight.callsign}</h3>
              <p>Altitude: {selectedFlight.altitude}ft</p>
              <p>Heading: {selectedFlight.heading}°</p>
              <p>Speed: {selectedFlight.velocity} kts</p>
              <p>Position: {selectedFlight.latitude.toFixed(4)}, {selectedFlight.longitude.toFixed(4)}</p>
            </div>
          )}
        </div>

        {/* Right Panel */}
        <div className="w-96 bg-black bg-opacity-50 border-l border-radar-green flex flex-col">
          {/* ATC Phase Selector */}
          <ATCPhaseSelector
            activePhase={activePhase}
            onPhaseChange={setActivePhase}
            categorizedFlights={categorizedFlights}
            atcCommunications={atcCommunications}
          />

          {/* Control Panel */}
          <ControlPanel
            isLive={isLive}
            onToggleLive={toggleLiveMode}
            playbackTime={playbackTime}
            onPlaybackTimeChange={setPlaybackTime}
          />

          {/* Recommendation Cards */}
          <RecommendationCards
            conflicts={conflicts}
            onResolve={handleConflictResolution}
          />

          {/* Timeline Panel */}
          <TimelinePanel
            atcCommunications={atcCommunications}
            conflicts={conflicts}
            activePhase={activePhase}
          />
        </div>
      </div>
    </div>
  );
}

export default App;
