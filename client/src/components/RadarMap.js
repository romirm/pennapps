import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

const RadarMap = ({ flights, categorizedFlights, conflicts, selectedFlight, onFlightSelect, isLive, activePhase }) => {
  const svgRef = useRef();
  const [dimensions, setDimensions] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const updateDimensions = () => {
      if (svgRef.current) {
        const rect = svgRef.current.getBoundingClientRect();
        setDimensions({ width: rect.width, height: rect.height });
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  useEffect(() => {
    if (!dimensions.width || !dimensions.height) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();

    // Set up the map projection (focus on Chicago area for ORD)
    const projection = d3.geoMercator()
      .center([-87.6298, 41.8781]) // Chicago coordinates
      .scale(50000)
      .translate([dimensions.width / 2, dimensions.height / 2]);

    // Create radar sweep effect

    // Add radar circles
    const radarCircles = [10, 20, 30, 40, 50]; // NM circles
    radarCircles.forEach((radius, i) => {
      svg.append('circle')
        .attr('cx', dimensions.width / 2)
        .attr('cy', dimensions.height / 2)
        .attr('r', radius * 1852 / 1000 * 50000 / 111000) // Convert NM to pixels
        .attr('fill', 'none')
        .attr('stroke', '#00ff00')
        .attr('stroke-width', 1)
        .attr('opacity', 0.3);
      
      // Add distance labels
      svg.append('text')
        .attr('x', dimensions.width / 2 + radius * 1852 / 1000 * 50000 / 111000)
        .attr('y', dimensions.height / 2 - 5)
        .attr('fill', '#00ff00')
        .attr('font-size', '12px')
        .attr('opacity', 0.7)
        .text(`${radius}NM`);
    });

    // Add compass rose
    const compassGroup = svg.append('g')
      .attr('class', 'compass-rose')
      .attr('transform', `translate(${dimensions.width - 100}, 100)`);

    // Draw compass
    compassGroup.append('circle')
      .attr('r', 30)
      .attr('fill', 'none')
      .attr('stroke', '#00ff00')
      .attr('stroke-width', 2);

    // Add cardinal directions
    const directions = ['N', 'E', 'S', 'W'];
    directions.forEach((dir, i) => {
      const angle = (i * 90) * Math.PI / 180;
      const x = Math.cos(angle) * 25;
      const y = Math.sin(angle) * 25;
      
      compassGroup.append('text')
        .attr('x', x)
        .attr('y', y)
        .attr('text-anchor', 'middle')
        .attr('fill', '#00ff00')
        .attr('font-size', '14px')
        .attr('font-weight', 'bold')
        .text(dir);
    });

    // Draw flight trails
    const trailGroup = svg.append('g').attr('class', 'flight-trails');
    
    flights.forEach(flight => {
      if (flight.latitude && flight.longitude) {
        const [x, y] = projection([flight.longitude, flight.latitude]);
        
        // Draw trail (simplified - in real implementation, store history)
        trailGroup.append('circle')
          .attr('cx', x)
          .attr('cy', y)
          .attr('r', 2)
          .attr('fill', '#00ff88')
          .attr('opacity', 0.3);
      }
    });

    // Draw conflict zones
    conflicts.forEach(conflict => {
      if (conflict.flight1.latitude && conflict.flight1.longitude && 
          conflict.flight2.latitude && conflict.flight2.longitude) {
        const [x1, y1] = projection([conflict.flight1.longitude, conflict.flight1.latitude]);
        const [x2, y2] = projection([conflict.flight2.longitude, conflict.flight2.latitude]);
        
        // Draw conflict line
        svg.append('line')
          .attr('x1', x1)
          .attr('y1', y1)
          .attr('x2', x2)
          .attr('y2', y2)
          .attr('stroke', '#ff0044')
          .attr('stroke-width', 2)
          .attr('stroke-dasharray', '5,5')
          .attr('opacity', 0.8);
        
        // Draw conflict zone
        const midX = (x1 + x2) / 2;
        const midY = (y1 + y2) / 2;
        const radius = Math.sqrt(Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2)) / 2;
        
        svg.append('circle')
          .attr('cx', midX)
          .attr('cy', midY)
          .attr('r', radius)
          .attr('fill', '#ff0044')
          .attr('opacity', 0.2)
          .attr('class', 'conflict-zone');
      }
    });

    // Draw flights by category
    const flightGroup = svg.append('g').attr('class', 'flights');
    
    // Get flights to display based on active phase
    let flightsToDisplay = [];
    if (activePhase === 'all') {
      flightsToDisplay = flights;
    } else {
      flightsToDisplay = categorizedFlights[activePhase] || [];
    }
    
    flightsToDisplay.forEach(flight => {
      if (flight.latitude && flight.longitude) {
        const [x, y] = projection([flight.longitude, flight.latitude]);
        const isInConflict = conflicts.some(c => 
          c.flight1.icao24 === flight.icao24 || c.flight2.icao24 === flight.icao24
        );
        const isSelected = selectedFlight && selectedFlight.icao24 === flight.icao24;
        
        // Get color based on flight category
        let flightColor = '#00ff00'; // Default green
        let flightSize = 4;
        
        switch(flight.category) {
          case 'enRoute':
            flightColor = '#00aaff'; // Blue for en route
            flightSize = 5;
            break;
          case 'taxi':
            flightColor = '#ffaa00'; // Orange for taxi
            flightSize = 6;
            break;
          case 'incoming':
            flightColor = '#ff6600'; // Red-orange for incoming
            flightSize = 7;
            break;
          default:
            flightColor = '#00ff00'; // Default green
            flightSize = 4;
            break;
        }
        
        // Flight dot
        const dot = flightGroup.append('g')
          .attr('class', `flight-dot ${isInConflict ? 'conflict-dot' : ''}`)
          .attr('transform', `translate(${x}, ${y})`)
          .style('cursor', 'pointer')
          .on('click', () => onFlightSelect(flight));
        
        // Main dot with category-based styling
        dot.append('circle')
          .attr('r', isSelected ? flightSize + 3 : flightSize)
          .attr('fill', isInConflict ? '#ff0044' : flightColor)
          .attr('stroke', isSelected ? '#ffffff' : 'none')
          .attr('stroke-width', isSelected ? 2 : 0);
        
        // Category indicator ring
        if (flight.category === 'taxi') {
          dot.append('circle')
            .attr('r', flightSize + 2)
            .attr('fill', 'none')
            .attr('stroke', flightColor)
            .attr('stroke-width', 1)
            .attr('opacity', 0.7);
        }
        
        // Heading indicator
        if (flight.heading) {
          const headingRad = (flight.heading - 90) * Math.PI / 180;
          const headX = Math.cos(headingRad) * (flightSize + 2);
          const headY = Math.sin(headingRad) * (flightSize + 2);
          
          dot.append('line')
            .attr('x1', 0)
            .attr('y1', 0)
            .attr('x2', headX)
            .attr('y2', headY)
            .attr('stroke', isInConflict ? '#ff0044' : flightColor)
            .attr('stroke-width', 2);
        }
        
        // Callsign label
        if (isSelected || isInConflict || flight.category === 'taxi') {
          dot.append('text')
            .attr('x', 12)
            .attr('y', 4)
            .attr('fill', flightColor)
            .attr('font-size', '12px')
            .attr('font-weight', 'bold')
            .text(flight.callsign);
        }
      }
    });

    // Add radar sweep animation
    if (isLive) {
      const sweep = svg.append('g')
        .attr('class', 'radar-sweep')
        .attr('transform', `translate(${dimensions.width / 2}, ${dimensions.height / 2})`);
      
      sweep.append('path')
        .attr('d', `M 0,0 L ${dimensions.width},0`)
        .attr('stroke', '#00ff00')
        .attr('stroke-width', 2)
        .attr('opacity', 0.8)
        .style('animation', 'radar-sweep 3s linear infinite');
    }

  }, [flights, categorizedFlights, conflicts, selectedFlight, dimensions, isLive, activePhase, onFlightSelect]);

  return (
    <div className="radar-container">
      <svg
        ref={svgRef}
        width="100%"
        height="100%"
        style={{ background: 'radial-gradient(circle, #001100 0%, #000000 100%)' }}
      />
    </div>
  );
};

export default RadarMap;
