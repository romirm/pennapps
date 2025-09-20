/**
 * JavaScript Integration Example for Bottleneck Predictions
 * 
 * This shows you how to fetch bottleneck predictions and display them
 * on your existing airport map visualization.
 */

// Function to fetch bottleneck predictions from your Flask API
async function fetchBottleneckPredictions(airportCode) {
    try {
        console.log(`🔍 Fetching bottleneck predictions for ${airportCode}...`);
        
        const response = await fetch(`/api/bottlenecks/${airportCode}`);
        const data = await response.json();
        
        if (data.success) {
            console.log(`✅ Found ${data.bottlenecks.length} bottlenecks`);
            return data;
        } else {
            console.error(`❌ Error: ${data.error}`);
            return null;
        }
    } catch (error) {
        console.error(`❌ Network error: ${error}`);
        return null;
    }
}

// Function to display bottleneck predictions on your map
function displayBottlenecksOnMap(bottlenecks, airportCoords) {
    console.log(`🗺️ Displaying ${bottlenecks.length} bottlenecks on map`);
    
    bottlenecks.forEach((bottleneck, index) => {
        const coordinates = bottleneck.location.coordinates;
        const zoneType = bottleneck.type;
        const probability = bottleneck.probability;
        const severity = bottleneck.severity;
        
        // Create bottleneck marker
        const marker = createBottleneckMarker(coordinates, zoneType, probability, severity);
        
        // Add to map
        addMarkerToMap(marker);
        
        // Log details
        console.log(`🚨 Bottleneck ${index + 1}: ${zoneType}`);
        console.log(`   📍 Location: ${coordinates[0]}, ${coordinates[1]}`);
        console.log(`   📊 Probability: ${probability.toFixed(2)}`);
        console.log(`   ⚠️ Severity: ${severity}/5`);
    });
}

// Function to create a bottleneck marker
function createBottleneckMarker(coordinates, zoneType, probability, severity) {
    // Determine color based on severity
    let color = '#00ff00'; // Green for low
    if (severity >= 4) color = '#ff0000';      // Red for high
    else if (severity >= 3) color = '#ff8800'; // Orange for medium-high
    else if (severity >= 2) color = '#ffff00'; // Yellow for medium
    
    // Determine size based on probability
    const size = Math.max(10, probability * 30);
    
    return {
        type: 'bottleneck',
        coordinates: coordinates,
        zoneType: zoneType,
        probability: probability,
        severity: severity,
        color: color,
        size: size,
        label: `${zoneType.toUpperCase()}\n${(probability * 100).toFixed(0)}%`
    };
}

// Function to add marker to your existing map
function addMarkerToMap(marker) {
    // This integrates with your existing map drawing code
    const canvas = document.getElementById('airportCanvas');
    const ctx = canvas.getContext('2d');
    
    // Convert coordinates to canvas position
    const canvasPos = convertCoordinatesToCanvas(marker.coordinates, canvas);
    
    // Draw bottleneck marker
    ctx.save();
    
    // Draw outer circle
    ctx.strokeStyle = marker.color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(canvasPos.x, canvasPos.y, marker.size, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Draw inner circle
    ctx.fillStyle = marker.color;
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    ctx.arc(canvasPos.x, canvasPos.y, marker.size * 0.6, 0, 2 * Math.PI);
    ctx.fill();
    
    // Draw label
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = '#ffffff';
    ctx.font = 'bold 12px Arial';
    ctx.textAlign = 'center';
    ctx.fillText(marker.label, canvasPos.x, canvasPos.y + 4);
    
    ctx.restore();
}

// Function to convert lat/lon coordinates to canvas position
function convertCoordinatesToCanvas(coordinates, canvas) {
    // This should match your existing coordinate conversion logic
    const airportCoords = [40.6398, -73.7789]; // JFK example
    const latDiff = coordinates[0] - airportCoords[0];
    const lonDiff = coordinates[1] - airportCoords[1];
    
    // Scale factors (adjust to match your map scaling)
    const scaleX = 100000; // Adjust based on your map
    const scaleY = 100000; // Adjust based on your map
    
    return {
        x: (canvas.width / 2) + (lonDiff * scaleX),
        y: (canvas.height / 2) - (latDiff * scaleY)
    };
}

// Function to update bottleneck panel in your sidebar
function updateBottleneckPanel(bottlenecks, summary) {
    const panel = document.getElementById('bottleneck-panel');
    if (!panel) return;
    
    // Clear existing content
    panel.innerHTML = '';
    
    // Add summary
    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'bottleneck-summary';
    summaryDiv.innerHTML = `
        <h3>🚨 Bottleneck Analysis</h3>
        <p><strong>Total Bottlenecks:</strong> ${summary.total_bottlenecks_predicted}</p>
        <p><strong>Risk Level:</strong> <span class="risk-${summary.overall_delay_risk}">${summary.overall_delay_risk.toUpperCase()}</span></p>
        <p><strong>Passengers at Risk:</strong> ${summary.total_passengers_at_risk}</p>
        <p><strong>Fuel Waste:</strong> ${summary.total_fuel_waste_estimate.toFixed(1)} gallons</p>
    `;
    panel.appendChild(summaryDiv);
    
    // Add individual bottlenecks
    bottlenecks.forEach((bottleneck, index) => {
        const bottleneckDiv = document.createElement('div');
        bottleneckDiv.className = 'bottleneck-item';
        bottleneckDiv.innerHTML = `
            <div class="bottleneck-header">
                <span class="bottleneck-type">${bottleneck.type.replace('_', ' ').toUpperCase()}</span>
                <span class="bottleneck-probability">${(bottleneck.probability * 100).toFixed(0)}%</span>
            </div>
            <div class="bottleneck-details">
                <p>Severity: ${bottleneck.severity}/5</p>
                <p>Delay: ${bottleneck.timing.estimated_duration_minutes.toFixed(1)} min</p>
                <p>Passengers: ${bottleneck.impact_analysis.passengers_affected}</p>
                <p>Fuel Waste: ${bottleneck.impact_analysis.fuel_waste_gallons.toFixed(1)} gal</p>
            </div>
        `;
        panel.appendChild(bottleneckDiv);
    });
}

// Main function to run bottleneck analysis
async function runBottleneckAnalysis(airportCode) {
    console.log(`🚀 Starting bottleneck analysis for ${airportCode}`);
    
    // 1. Fetch bottleneck predictions
    const data = await fetchBottleneckPredictions(airportCode);
    
    if (!data) {
        console.error('❌ Failed to get bottleneck predictions');
        return;
    }
    
    // 2. Display bottlenecks on map
    const airportCoords = [40.6398, -73.7789]; // Get from your airport data
    displayBottlenecksOnMap(data.bottlenecks, airportCoords);
    
    // 3. Update sidebar panel
    updateBottleneckPanel(data.bottlenecks, data.summary);
    
    console.log(`✅ Bottleneck analysis complete!`);
}

// Integration with your existing airport page
document.addEventListener('DOMContentLoaded', function() {
    // Get airport code from URL or page data
    const airportCode = window.location.pathname.split('/').pop().toUpperCase();
    
    if (airportCode && airportCode !== '') {
        // Run bottleneck analysis when page loads
        runBottleneckAnalysis(airportCode);
        
        // Set up periodic updates (every 30 seconds)
        setInterval(() => {
            runBottleneckAnalysis(airportCode);
        }, 30000);
    }
});

// Add CSS styles for bottleneck display
const bottleneckStyles = `
<style>
.bottleneck-summary {
    background: #1a1a1a;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 15px;
}

.bottleneck-summary h3 {
    color: #ff6b6b;
    margin: 0 0 10px 0;
}

.bottleneck-summary p {
    margin: 5px 0;
    color: #ffffff;
}

.risk-critical { color: #ff0000; font-weight: bold; }
.risk-high { color: #ff8800; font-weight: bold; }
.risk-medium { color: #ffff00; font-weight: bold; }
.risk-low { color: #00ff00; font-weight: bold; }

.bottleneck-item {
    background: #2a2a2a;
    padding: 10px;
    border-radius: 5px;
    margin-bottom: 10px;
    border-left: 4px solid #ff6b6b;
}

.bottleneck-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
}

.bottleneck-type {
    color: #ff6b6b;
    font-weight: bold;
    font-size: 14px;
}

.bottleneck-probability {
    background: #ff6b6b;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: bold;
}

.bottleneck-details p {
    margin: 3px 0;
    color: #cccccc;
    font-size: 12px;
}
</style>
`;

// Add styles to page
document.head.insertAdjacentHTML('beforeend', bottleneckStyles);
