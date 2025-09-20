import React from 'react';

const RecommendationCards = ({ conflicts, onResolve }) => {
  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'High': return 'text-red-400 border-red-400 bg-red-900 bg-opacity-20';
      case 'Medium': return 'text-yellow-400 border-yellow-400 bg-yellow-900 bg-opacity-20';
      case 'Low': return 'text-green-400 border-green-400 bg-green-900 bg-opacity-20';
      default: return 'text-gray-400 border-gray-400 bg-gray-900 bg-opacity-20';
    }
  };

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'High': return '🚨';
      case 'Medium': return '⚠️';
      case 'Low': return 'ℹ️';
      default: return '📋';
    }
  };

  if (conflicts.length === 0) {
    return (
      <div className="p-4 border-b border-radar-green">
        <h3 className="text-lg font-bold mb-2">AI Recommendations</h3>
        <div className="text-center text-gray-400 py-8">
          <div className="text-4xl mb-2">✅</div>
          <p>No conflicts detected</p>
          <p className="text-sm">All aircraft maintaining safe separation</p>
        </div>
      </div>
    );
  }

  return (
    <div className="p-4 border-b border-radar-green max-h-96 overflow-y-auto">
      <h3 className="text-lg font-bold mb-4">AI Recommendations</h3>
      <div className="space-y-3">
        {conflicts.map((conflict) => (
          <div
            key={conflict.id}
            className={`recommendation-card p-4 rounded border-2 ${getRiskColor(conflict.riskLevel)}`}
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center space-x-2">
                <span className="text-xl">{getRiskIcon(conflict.riskLevel)}</span>
                <span className="font-bold">{conflict.riskLevel} Risk</span>
              </div>
              <span className="text-xs opacity-75">
                {new Date(conflict.timestamp).toLocaleTimeString()}
              </span>
            </div>
            
            <div className="mb-3">
              <p className="text-sm font-semibold mb-1">
                {conflict.flight1.callsign} ↔ {conflict.flight2.callsign}
              </p>
              <p className="text-xs opacity-75">
                Distance: {conflict.distance.toFixed(2)}km | 
                Alt Diff: {conflict.altitudeDifference.toFixed(0)}ft
              </p>
            </div>

            {conflict.aiRecommendation && (
              <div className="mb-3 p-2 bg-black bg-opacity-30 rounded">
                <p className="text-sm font-semibold mb-1">AI Analysis:</p>
                <p className="text-xs mb-1">{conflict.aiRecommendation.summary}</p>
                <p className="text-xs mb-1">
                  <strong>Recommendation:</strong> {conflict.aiRecommendation.recommendation}
                </p>
                <p className="text-xs">
                  <strong>Action:</strong> {conflict.aiRecommendation.suggestedAction}
                </p>
              </div>
            )}

            <div className="flex space-x-2">
              <button
                onClick={() => onResolve(conflict.id, 'accept')}
                className="flex-1 px-3 py-1 bg-green-600 text-white text-xs rounded hover:bg-green-700 transition-colors"
              >
                ✅ Accept
              </button>
              <button
                onClick={() => onResolve(conflict.id, 'reject')}
                className="flex-1 px-3 py-1 bg-red-600 text-white text-xs rounded hover:bg-red-700 transition-colors"
              >
                ❌ Reject
              </button>
              <button
                onClick={() => onResolve(conflict.id, 'edit')}
                className="flex-1 px-3 py-1 bg-blue-600 text-white text-xs rounded hover:bg-blue-700 transition-colors"
              >
                ✏️ Edit
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default RecommendationCards;
