import React from 'react';

const ATCPhaseSelector = ({ activePhase, onPhaseChange, categorizedFlights, atcCommunications }) => {
  const phases = [
    { key: 'all', label: 'All Phases', color: '#00ff00', count: Object.values(categorizedFlights).flat().length },
    { key: 'enRoute', label: 'En Route', color: '#00aaff', count: categorizedFlights.enRoute?.length || 0 },
    { key: 'taxi', label: 'Taxi', color: '#ffaa00', count: categorizedFlights.taxi?.length || 0 },
    { key: 'incoming', label: 'Incoming', color: '#ff6600', count: categorizedFlights.incoming?.length || 0 }
  ];

  return (
    <div className="p-4 border-b border-radar-green">
      <h3 className="text-lg font-bold mb-4">ATC Phases</h3>
      
      <div className="space-y-2">
        {phases.map(phase => (
          <button
            key={phase.key}
            onClick={() => onPhaseChange(phase.key)}
            className={`w-full p-3 rounded border-2 transition-all ${
              activePhase === phase.key
                ? 'border-white bg-white bg-opacity-20'
                : 'border-gray-600 hover:border-gray-400'
            }`}
            style={{ borderColor: phase.color }}
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-3">
                <div 
                  className="w-4 h-4 rounded-full"
                  style={{ backgroundColor: phase.color }}
                ></div>
                <span className="font-semibold">{phase.label}</span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-sm opacity-75">{phase.count} flights</span>
                {atcCommunications[phase.key]?.length > 0 && (
                  <span className="text-xs bg-blue-600 text-white px-2 py-1 rounded">
                    {atcCommunications[phase.key].length} comms
                  </span>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>
      
      {/* Phase Legend */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <h4 className="text-sm font-semibold mb-2">Phase Legend</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
            <span>En Route - High altitude, long distance</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-orange-400 rounded-full"></div>
            <span>Taxi - On ground, near airport</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-400 rounded-full"></div>
            <span>Incoming - Descending, approaching</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ATCPhaseSelector;
