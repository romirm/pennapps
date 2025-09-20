import React, { useState, useEffect } from 'react';

const TimelinePanel = ({ atcCommunications, conflicts, activePhase }) => {
  const [timelineItems, setTimelineItems] = useState([]);

  useEffect(() => {
    // Combine different data sources into timeline
    const items = [];
    
    // Add conflicts as timeline items
    conflicts.forEach(conflict => {
      items.push({
        id: `conflict-${conflict.id}`,
        timestamp: new Date(conflict.timestamp),
        type: 'conflict',
        title: `Conflict Detected: ${conflict.flight1.callsign} ↔ ${conflict.flight2.callsign}`,
        description: `Distance: ${conflict.distance.toFixed(2)}km, Alt Diff: ${conflict.altitudeDifference.toFixed(0)}ft`,
        riskLevel: conflict.riskLevel,
        aiRecommendation: conflict.aiRecommendation
      });
    });

    // Add ATC communications based on active phase
    if (activePhase === 'all') {
      // Show all communications
      Object.keys(atcCommunications).forEach(phase => {
        atcCommunications[phase].forEach(comm => {
          items.push({
            ...comm,
            phase: phase
          });
        });
      });
    } else {
      // Show only active phase communications
      if (atcCommunications[activePhase]) {
        atcCommunications[activePhase].forEach(comm => {
          items.push({
            ...comm,
            phase: activePhase
          });
        });
      }
    }

    // Sort by timestamp
    items.sort((a, b) => b.timestamp - a.timestamp);
    setTimelineItems(items);
  }, [conflicts, atcCommunications, activePhase]);

  const getTypeIcon = (type) => {
    switch (type) {
      case 'conflict': return '🚨';
      case 'atc': return '🎧';
      case 'pilot': return '✈️';
      case 'ai': return '🤖';
      default: return '📋';
    }
  };

  const getTypeColor = (type, phase) => {
    if (type === 'conflict') return 'border-red-400 text-red-400';
    if (type === 'ai') return 'border-purple-400 text-purple-400';
    
    // Color based on phase for ATC communications
    switch (phase) {
      case 'enRoute': return 'border-blue-400 text-blue-400';
      case 'taxi': return 'border-orange-400 text-orange-400';
      case 'incoming': return 'border-red-400 text-red-400';
      default: return 'border-gray-400 text-gray-400';
    }
  };


  return (
    <div className="flex-1 p-4 overflow-y-auto">
      <h3 className="text-lg font-bold mb-4">Live Timeline</h3>
      
      {timelineItems.length === 0 ? (
        <div className="text-center text-gray-400 py-8">
          <div className="text-4xl mb-2">📡</div>
          <p>No activity detected</p>
          <p className="text-sm">Timeline will populate as events occur</p>
        </div>
      ) : (
        <div className="space-y-3">
          {timelineItems.map((item) => (
            <div
              key={item.id}
              className={`timeline-item p-3 rounded border-l-4 ${getTypeColor(item.type, item.phase)}`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center space-x-2">
                  <span className="text-lg">{getTypeIcon(item.type)}</span>
                  <span className="font-semibold text-sm">{item.title || item.speaker}</span>
                </div>
                <span className="text-xs opacity-75">
                  {item.timestamp.toLocaleTimeString()}
                </span>
              </div>
              
              <p className="text-sm mb-2">{item.message || item.description}</p>
              
              {item.callsign && (
                <span className="inline-block px-2 py-1 bg-black bg-opacity-30 rounded text-xs">
                  {item.callsign}
                </span>
              )}
              
              {item.riskLevel && (
                <span className={`inline-block px-2 py-1 rounded text-xs ml-2 ${
                  item.riskLevel === 'High' ? 'bg-red-900 text-red-300' :
                  item.riskLevel === 'Medium' ? 'bg-yellow-900 text-yellow-300' :
                  'bg-green-900 text-green-300'
                }`}>
                  {item.riskLevel} Risk
                </span>
              )}
              
              {item.aiRecommendation && (
                <div className="mt-2 p-2 bg-black bg-opacity-30 rounded">
                  <p className="text-xs font-semibold mb-1">AI Recommendation:</p>
                  <p className="text-xs">{item.aiRecommendation.recommendation}</p>
                </div>
              )}
              
              {item.recommendation && (
                <div className="mt-2 p-2 bg-purple-900 bg-opacity-30 rounded">
                  <p className="text-xs font-semibold mb-1">AI Analysis:</p>
                  <p className="text-xs">{item.recommendation}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

export default TimelinePanel;
