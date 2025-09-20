# ATC AI Co-Pilot Dashboard - Demo Guide

## 🚀 Quick Start Demo

### 1. Start the Application
```bash
# Windows
start.bat

# Mac/Linux
./start.sh

# Or manually
npm run install-all
npm run dev
```

### 2. Access the Dashboard
- Open your browser to `http://localhost:3000`
- Backend API runs on `http://localhost:5000`

## 🎯 Demo Flow

### Step 1: Initial Load
- Dashboard loads with radar-style interface
- Shows flights around O'Hare (ORD) area
- Real-time data from OpenSky Network (or demo data if unavailable)

### Step 2: Explore the Interface
- **Radar Map**: Central view with flight positions
- **Control Panel**: Live/Playback toggle and system status
- **Recommendation Cards**: AI conflict alerts (if any)
- **Timeline Panel**: Live ATC communications and events

### Step 3: Interact with Flights
- Click on any flight dot to see details
- Hover over flights for quick info
- Watch the radar sweep animation (live mode)

### Step 4: Monitor AI Recommendations
- AI automatically detects potential conflicts
- Review recommendation cards in the right panel
- Test Accept/Reject/Edit buttons

### Step 5: Timeline View
- Scroll through live ATC communications
- See AI analysis and recommendations
- Color-coded events (ATC, Pilot, AI, Conflicts)

## 🎨 Key Features to Highlight

### Visual Elements
- **Radar Aesthetic**: Dark background with neon-green elements
- **Animated Sweep**: Rotating radar sweep effect
- **Distance Rings**: 10, 20, 30, 40, 50 NM circles
- **Compass Rose**: Cardinal directions indicator
- **Conflict Zones**: Red highlighting for potential conflicts

### Interactive Features
- **Real-time Updates**: Data refreshes every 5 seconds
- **WebSocket Communication**: Live data streaming
- **Conflict Detection**: 3NM/1000ft separation rules
- **AI Recommendations**: Intelligent controller suggestions

### Technical Highlights
- **OpenSky Integration**: Live ADS-B flight data
- **Cerebras AI**: LLM-powered conflict analysis
- **D3.js Visualization**: Smooth flight path rendering
- **Responsive Design**: Modern UI with Tailwind CSS

## 🔧 Demo Customization

### For Different Airports
Edit `server/index.js` line 149:
```javascript
const projection = d3.geoMercator()
  .center([-87.6298, 41.8781]) // Change to your airport coordinates
  .scale(50000)
```

### For More Conflicts
Edit `server/demoData.js` to generate more conflicts:
```javascript
const numConflicts = Math.floor(Math.random() * 3) + 2; // More conflicts
```

### For Different AI Responses
Add your Cerebras API key to `server/.env`:
```
CEREBRAS_API_KEY=your_key_here
```

## 🎪 Demo Script

### Opening (30 seconds)
1. "Welcome to the ATC AI Co-Pilot Dashboard"
2. "This is a real-time flight tracking system with AI-powered conflict detection"
3. "We're monitoring flights around O'Hare International Airport"

### Core Demo (2 minutes)
1. "Here's our radar-style interface showing live flight data"
2. "The AI automatically detects potential conflicts using 3NM/1000ft separation rules"
3. "When conflicts are detected, we get intelligent recommendations"
4. "Controllers can accept, reject, or edit these suggestions"
5. "The timeline shows all ATC communications and AI analysis"

### Technical Deep Dive (1 minute)
1. "Data comes from OpenSky Network's free ADS-B API"
2. "AI analysis powered by Cerebras API for intelligent recommendations"
3. "Real-time updates via WebSocket for smooth user experience"
4. "Built with React, D3.js, and Node.js for modern performance"

### Closing (30 seconds)
1. "This system could revolutionize air traffic control"
2. "AI assistance reduces controller workload and improves safety"
3. "Questions?"

## 🐛 Troubleshooting

### Common Issues
- **No flights showing**: Check OpenSky API status, demo data will load
- **WebSocket errors**: Ensure backend is running on port 5000
- **Styling issues**: Run `npm install` in client directory
- **Port conflicts**: Change ports in package.json scripts

### Performance Tips
- Close other browser tabs for better performance
- Use Chrome/Firefox for best D3.js rendering
- Ensure stable internet connection for OpenSky API

## 📊 Demo Data

The system includes realistic demo data:
- 8-12 mock flights around O'Hare
- 1-2 simulated conflicts
- Realistic ATC communications
- AI-generated recommendations

## 🎯 Wow Factor

- **Live Data**: Real flights from OpenSky Network
- **AI Integration**: Cerebras API for intelligent analysis
- **Professional UI**: Authentic ATC radar aesthetic
- **Real-time**: WebSocket updates every 5 seconds
- **Interactive**: Click, hover, and control elements

---

**Ready to demo! 🚀**
