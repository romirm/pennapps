"""
Bottleneck Visualization Integration

This script demonstrates how to integrate bottleneck predictions
with the airport visualization app to show real-time bottleneck
locations on the airport layout.
"""

import json
import math
from typing import Dict, List, Tuple


class BottleneckVisualizer:
    """
    Integrates bottleneck predictions with airport visualization
    """
    
    def __init__(self):
        # Bottleneck zone definitions with visual properties
        self.bottleneck_zones = {
            'runway_approach': {
                'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.001,
                'color': '#ff4444', 'opacity': 0.7, 'priority': 'high'
            },
            'runway_departure': {
                'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.0008,
                'color': '#ff8844', 'opacity': 0.6, 'priority': 'high'
            },
            'taxiway_intersection': {
                'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.0005,
                'color': '#ffaa44', 'opacity': 0.8, 'priority': 'medium'
            },
            'gate_area': {
                'center_lat': 40.6413, 'center_lon': -73.7781, 'radius': 0.002,
                'color': '#ffcc44', 'opacity': 0.5, 'priority': 'low'
            }
        }
    
    def convert_to_canvas_coordinates(self, lat: float, lon: float, canvas_width: int, canvas_height: int) -> Tuple[int, int]:
        """Convert lat/lon to canvas pixel coordinates"""
        # JFK approximate bounds
        min_lat, max_lat = 40.635, 40.650
        min_lon, max_lon = -73.785, -73.770
        
        # Normalize coordinates
        x = (lon - min_lon) / (max_lon - min_lon)
        y = (lat - min_lat) / (max_lat - min_lat)
        
        # Convert to canvas coordinates
        canvas_x = int(x * canvas_width)
        canvas_y = int((1 - y) * canvas_height)  # Flip Y axis
        
        return canvas_x, canvas_y
    
    def generate_visualization_data(self, bottleneck_analysis: Dict, canvas_width: int = 1200, canvas_height: int = 800) -> Dict:
        """Generate visualization data for the frontend"""
        
        visualization_data = {
            'canvas_width': canvas_width,
            'canvas_height': canvas_height,
            'bottlenecks': [],
            'aircraft_positions': [],
            'impact_summary': bottleneck_analysis.get('impact_metrics', {}),
            'airport_summary': bottleneck_analysis.get('airport_summary', {})
        }
        
        # Process bottleneck predictions
        for bottleneck in bottleneck_analysis.get('bottleneck_predictions', []):
            zone_type = bottleneck['type']
            coords = bottleneck['location']['coordinates']
            lat, lon = coords[0], coords[1]
            
            # Convert to canvas coordinates
            canvas_x, canvas_y = self.convert_to_canvas_coordinates(lat, lon, canvas_width, canvas_height)
            
            # Get zone properties
            zone_props = self.bottleneck_zones.get(zone_type, self.bottleneck_zones['taxiway_intersection'])
            
            # Calculate visual properties based on severity
            severity = bottleneck['severity']
            intensity = min(severity / 5.0, 1.0)
            
            bottleneck_visual = {
                'id': bottleneck['bottleneck_id'],
                'type': zone_type,
                'canvas_x': canvas_x,
                'canvas_y': canvas_y,
                'radius_pixels': int(zone_props['radius'] * 100000),  # Convert to pixels
                'color': zone_props['color'],
                'opacity': zone_props['opacity'] * intensity,
                'severity': severity,
                'probability': bottleneck['probability'],
                'aircraft_count': len(bottleneck['aircraft_affected']),
                'passengers_affected': bottleneck['impact_analysis']['passengers_affected'],
                'fuel_waste': bottleneck['impact_analysis']['fuel_waste_gallons'],
                'economic_impact': bottleneck['impact_analysis']['economic_impact_estimate'],
                'recommendations': bottleneck['recommended_mitigations'][:2]  # Top 2 recommendations
            }
            
            visualization_data['bottlenecks'].append(bottleneck_visual)
        
        # Process aircraft positions
        for aircraft in bottleneck_analysis.get('aircraft_affected', []):
            # This would come from the original ADS-B data
            # For now, we'll use the bottleneck analysis data
            pass
        
        return visualization_data
    
    def generate_javascript_code(self, visualization_data: Dict) -> str:
        """Generate JavaScript code to render bottlenecks on the canvas"""
        
        js_code = f"""
// Bottleneck Visualization Code
const bottleneckData = {json.dumps(visualization_data, indent=2)};

function drawBottlenecks() {{
    const ctx = canvas.getContext('2d');
    
    // Draw each bottleneck
    bottleneckData.bottlenecks.forEach(bottleneck => {{
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
    }});
}}

function drawBottleneckInfo(bottleneck, x, y) {{
    const ctx = canvas.getContext('2d');
    const infoY = y - bottleneck.radius_pixels - 30;
    
    // Background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
    ctx.fillRect(x - 100, infoY - 20, 200, 40);
    
    // Text
    ctx.fillStyle = '#ffffff';
    ctx.font = '12px JetBrains Mono';
    ctx.textAlign = 'center';
    ctx.fillText(`${{bottleneck.type.replace('_', ' ').toUpperCase()}}`, x, infoY);
    ctx.fillText(`${{bottleneck.aircraft_count}} aircraft, ${{bottleneck.passengers_affected}} passengers`, x, infoY + 15);
}}

function updateBottleneckDisplay() {{
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
}}

function updateBottleneckInfoPanel() {{
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
}}

// Initialize bottleneck visualization
updateBottleneckDisplay();
"""
        
        return js_code
    
    def generate_html_integration(self, visualization_data: Dict) -> str:
        """Generate HTML code to integrate with the airport app"""
        
        html_code = f"""
<!-- Bottleneck Visualization Integration -->
<div id="bottleneck-panel" class="bottleneck-panel">
    <h3>🚨 Bottleneck Analysis</h3>
    
    <div class="bottleneck-summary">
        <div class="summary-item">
            <span class="label">Bottlenecks:</span>
            <span class="value" id="bottleneck-count">{visualization_data['airport_summary']['total_bottlenecks_predicted']}</span>
        </div>
        <div class="summary-item">
            <span class="label">Severity:</span>
            <span class="value" id="severity-level">{visualization_data['airport_summary']['highest_severity_level']}/5</span>
        </div>
        <div class="summary-item">
            <span class="label">Risk Level:</span>
            <span class="value" id="delay-risk">{visualization_data['airport_summary']['overall_delay_risk'].upper()}</span>
        </div>
        <div class="summary-item">
            <span class="label">Passengers at Risk:</span>
            <span class="value" id="passengers-at-risk">{visualization_data['airport_summary']['total_passengers_at_risk']}</span>
        </div>
        <div class="summary-item">
            <span class="label">Fuel Waste:</span>
            <span class="value" id="fuel-waste">{visualization_data['airport_summary']['total_fuel_waste_estimate']:.1f} gal</span>
        </div>
    </div>
    
    <div class="impact-metrics">
        <h4>💰 Impact Analysis</h4>
        <div class="metric-item">
            <span class="label">Economic Impact:</span>
            <span class="value" id="economic-impact">${visualization_data['impact_summary']['total_economic_impact_usd']:.2f}</span>
        </div>
        <div class="metric-item">
            <span class="label">CO2 Emissions:</span>
            <span class="value" id="co2-emissions">{visualization_data['impact_summary']['total_co2_emissions_lbs']:.1f} lbs</span>
        </div>
    </div>
    
    <div class="bottleneck-details">
        <h4>🚨 Active Bottlenecks</h4>
        <div id="bottleneck-list">
            <!-- Bottleneck details will be populated here -->
        </div>
    </div>
</div>

<style>
.bottleneck-panel {{
    background: rgba(15, 15, 15, 0.95);
    border: 1px solid #333;
    border-radius: 8px;
    padding: 20px;
    margin: 20px;
    color: #e0e0e0;
    font-family: 'JetBrains Mono', monospace;
    max-width: 300px;
}}

.bottleneck-summary {{
    margin-bottom: 20px;
}}

.summary-item {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 8px;
    padding: 5px 0;
    border-bottom: 1px solid #333;
}}

.summary-item .label {{
    color: #999;
    font-size: 12px;
}}

.summary-item .value {{
    color: #00ff00;
    font-weight: bold;
    font-size: 14px;
}}

.impact-metrics {{
    margin-bottom: 20px;
}}

.impact-metrics h4 {{
    color: #ffaa00;
    margin-bottom: 10px;
    font-size: 14px;
}}

.metric-item {{
    display: flex;
    justify-content: space-between;
    margin-bottom: 5px;
}}

.metric-item .label {{
    color: #ccc;
    font-size: 11px;
}}

.metric-item .value {{
    color: #ffaa00;
    font-weight: bold;
    font-size: 12px;
}}

.bottleneck-details h4 {{
    color: #ff4444;
    margin-bottom: 10px;
    font-size: 14px;
}}

#bottleneck-list {{
    max-height: 200px;
    overflow-y: auto;
}}

.bottleneck-item {{
    background: rgba(255, 68, 68, 0.1);
    border: 1px solid #ff4444;
    border-radius: 4px;
    padding: 10px;
    margin-bottom: 10px;
}}

.bottleneck-item .type {{
    color: #ff4444;
    font-weight: bold;
    font-size: 12px;
}}

.bottleneck-item .details {{
    color: #ccc;
    font-size: 11px;
    margin-top: 5px;
}}
</style>
"""
        
        return html_code


def main():
    """Demonstrate bottleneck visualization integration"""
    print("🎨 Bottleneck Visualization Integration Demo")
    print("=" * 50)
    
    # Load previous analysis results
    try:
        with open('simplified_bottleneck_analysis_20250920_134414.json', 'r') as f:
            analysis = json.load(f)
        print("✅ Loaded bottleneck analysis results")
    except FileNotFoundError:
        print("❌ No analysis results found. Run simplified_bottleneck_demo.py first.")
        return
    
    # Create visualizer
    visualizer = BottleneckVisualizer()
    
    # Generate visualization data
    print("\n🎯 Generating visualization data...")
    viz_data = visualizer.generate_visualization_data(analysis)
    
    print(f"✅ Generated visualization data:")
    print(f"   - {len(viz_data['bottlenecks'])} bottlenecks to visualize")
    print(f"   - Canvas size: {viz_data['canvas_width']}x{viz_data['canvas_height']}")
    print(f"   - Total economic impact: ${viz_data['impact_summary']['total_economic_impact_usd']:.2f}")
    
    # Generate JavaScript code
    print("\n📝 Generating JavaScript integration code...")
    js_code = visualizer.generate_javascript_code(viz_data)
    
    # Save JavaScript code
    with open('bottleneck_visualization.js', 'w') as f:
        f.write(js_code)
    
    # Generate HTML integration
    print("\n🌐 Generating HTML integration code...")
    html_code = visualizer.generate_html_integration(viz_data)
    
    # Save HTML code
    with open('bottleneck_panel.html', 'w', encoding='utf-8') as f:
        f.write(html_code)
    
    # Save visualization data
    with open('bottleneck_visualization_data.json', 'w') as f:
        json.dump(viz_data, f, indent=2)
    
    print("\n✅ Integration files created:")
    print("   - bottleneck_visualization.js (JavaScript code)")
    print("   - bottleneck_panel.html (HTML panel)")
    print("   - bottleneck_visualization_data.json (Data)")
    
    print("\n🎯 Next Steps:")
    print("1. Add the HTML panel to your airport.html template")
    print("2. Include the JavaScript code in your airport.html")
    print("3. Call updateBottleneckDisplay() after drawing the airport layout")
    print("4. The bottlenecks will appear as colored circles on the canvas")
    
    print("\n📊 Visualization Features:")
    print("✅ Real-time bottleneck overlay on airport layout")
    print("✅ Color-coded severity indicators")
    print("✅ Interactive info panels")
    print("✅ Economic impact display")
    print("✅ Mitigation recommendations")


if __name__ == "__main__":
    main()
