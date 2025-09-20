
// Bottleneck Visualization Code
const bottleneckData = {
  "canvas_width": 1200,
  "canvas_height": 800,
  "bottlenecks": [
    {
      "id": "taxiway_intersection",
      "type": "taxiway_intersection",
      "canvas_x": 552,
      "canvas_y": 463,
      "radius_pixels": 50,
      "color": "#ffaa44",
      "opacity": 0.8,
      "severity": 6,
      "probability": 1.0,
      "aircraft_count": 1,
      "passengers_affected": 120,
      "fuel_waste": 37.5,
      "economic_impact": 4631.25,
      "recommendations": [
        {
          "action": "Implement ground traffic sequencing",
          "priority": "medium",
          "estimated_effectiveness": 0.7,
          "implementation_time": 3.0
        },
        {
          "action": "Use alternative taxi routes",
          "priority": "low",
          "estimated_effectiveness": 0.5,
          "implementation_time": 2.0
        }
      ]
    }
  ],
  "aircraft_positions": [],
  "impact_summary": {
    "total_passengers_affected": 120,
    "total_fuel_waste_gallons": 37.5,
    "total_economic_impact_usd": 4631.25,
    "total_co2_emissions_lbs": 791.25,
    "average_delay_minutes": 15
  },
  "airport_summary": {
    "total_bottlenecks_predicted": 1,
    "highest_severity_level": 6,
    "total_passengers_at_risk": 120,
    "total_fuel_waste_estimate": 37.5,
    "overall_delay_risk": "critical"
  }
};

function drawBottlenecks() {
    const ctx = canvas.getContext('2d');
    
    // Draw each bottleneck
    bottleneckData.bottlenecks.forEach(bottleneck => {
        const x = bottleneck.canvas_x;
        const y = bottleneck.canvas_y;
        const radius = bottleneck.radius_pixels;
        
        // Draw bottleneck zone
        ctx.save();
        ctx.globalAlpha = bottleneck.opacity;
        ctx.fillStyle = bottleneck.color;
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, 2 * Math.PI);
        ctx.fill();
        
        // Draw border
        ctx.globalAlpha = 1.0;
        ctx.strokeStyle = bottleneck.color;
        ctx.lineWidth = 3;
        ctx.stroke();
        
        // Draw severity indicator
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 16px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.fillText(bottleneck.severity.toString(), x, y + 5);
        
        ctx.restore();
        
        // Draw bottleneck info
        drawBottleneckInfo(bottleneck, x, y);
    });
}

function drawBottleneckInfo(bottleneck, x, y) {
    const ctx = canvas.getContext('2d');
    const infoY = y - bottleneck.radius_pixels - 30;
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(x - 100, infoY - 20, 200, 40);
    
    // Text
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`${bottleneck.type.replace('_', ' ').toUpperCase()}`, x, infoY);
    ctx.fillText(`${bottleneck.aircraft_count} aircraft, ${bottleneck.passengers_affected} passengers`, x, infoY + 15);
}

function updateBottleneckDisplay() {
    // Clear previous bottlenecks
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Redraw airport layout
    drawTaxiways();
    drawRunways();
    drawScaleReference();
    
    // Draw bottlenecks
    drawBottlenecks();
    
    // Update info panel
    updateBottleneckInfoPanel();
}

function updateBottleneckInfoPanel() {
    const summary = bottleneckData.airport_summary;
    const impact = bottleneckData.impact_summary;
    
    // Update bottleneck count
    document.getElementById('bottleneck-count').textContent = summary.total_bottlenecks_predicted;
    document.getElementById('severity-level').textContent = summary.highest_severity_level;
    document.getElementById('delay-risk').textContent = summary.overall_delay_risk.toUpperCase();
    document.getElementById('passengers-at-risk').textContent = summary.total_passengers_at_risk;
    document.getElementById('fuel-waste').textContent = summary.total_fuel_waste_estimate.toFixed(1);
    
    // Update impact metrics
    document.getElementById('economic-impact').textContent = '$' + impact.total_economic_impact_usd.toFixed(2);
    document.getElementById('co2-emissions').textContent = impact.total_co2_emissions_lbs.toFixed(1);
}

// Initialize bottleneck visualization
updateBottleneckDisplay();
