import React from 'react';

const ControlPanel = ({ isLive, onToggleLive, playbackTime, onPlaybackTimeChange }) => {
  const handlePlaybackChange = (e) => {
    onPlaybackTimeChange(parseInt(e.target.value));
  };

  return (
    <div className="p-4 border-b border-radar-green">
      <h3 className="text-lg font-bold mb-4">Control Panel</h3>
      
      {/* Live/Playback Toggle */}
      <div className="mb-4">
        <div className="flex items-center space-x-4 mb-2">
          <button
            onClick={() => onToggleLive(true)}
            className={`control-button ${isLive ? 'active' : ''}`}
          >
            📡 Live Feed
          </button>
          <button
            onClick={() => onToggleLive(false)}
            className={`control-button ${!isLive ? 'active' : ''}`}
          >
            ⏯️ Playback
          </button>
        </div>
        
        {!isLive && (
          <div className="mt-3">
            <label className="block text-sm mb-2">Playback Time</label>
            <input
              type="range"
              min="0"
              max="100"
              value={playbackTime}
              onChange={handlePlaybackChange}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
            />
            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>00:00</span>
              <span>{Math.floor(playbackTime / 60)}:{(playbackTime % 60).toString().padStart(2, '0')}</span>
              <span>10:00</span>
            </div>
          </div>
        )}
      </div>

      {/* System Status */}
      <div className="mb-4">
        <h4 className="text-sm font-semibold mb-2">System Status</h4>
        <div className="space-y-1 text-xs">
          <div className="flex justify-between">
            <span>OpenSky API:</span>
            <span className="text-green-400">● Connected</span>
          </div>
          <div className="flex justify-between">
            <span>Cerebras AI:</span>
            <span className="text-green-400">● Active</span>
          </div>
          <div className="flex justify-between">
            <span>WebSocket:</span>
            <span className="text-green-400">● Live</span>
          </div>
          <div className="flex justify-between">
            <span>LiveATC:</span>
            <span className="text-yellow-400">○ Mock Data</span>
          </div>
        </div>
      </div>

      {/* Quick Actions */}
      <div>
        <h4 className="text-sm font-semibold mb-2">Quick Actions</h4>
        <div className="space-y-2">
          <button className="w-full control-button text-xs">
            🎯 Focus on ORD
          </button>
          <button className="w-full control-button text-xs">
            📊 Show Statistics
          </button>
          <button className="w-full control-button text-xs">
            🔄 Refresh Data
          </button>
          <button className="w-full control-button text-xs">
            📹 Record Session
          </button>
        </div>
      </div>

      {/* Legend */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <h4 className="text-sm font-semibold mb-2">Legend</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-400 rounded-full"></div>
            <span>Normal Flight</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-red-400 rounded-full"></div>
            <span>Conflict Alert</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-blue-400 rounded-full"></div>
            <span>Selected Flight</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-yellow-400 rounded-full"></div>
            <span>AI Recommendation</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ControlPanel;
