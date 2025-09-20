# ATC AI Co-Pilot Dashboard

A real-time interactive dashboard that streams live flight data and uses AI to detect conflicts and provide controller recommendations.

## 🚀 Features

### Core Functionality
- **Live Flight Tracking**: Real-time ADS-B data from OpenSky Network
- **AI Conflict Detection**: Automatic detection of potential conflicts (3NM/1000ft separation)
- **Radar-Style Visualization**: D3.js-powered map with authentic ATC radar aesthetics
- **AI Recommendations**: Cerebras API integration for intelligent controller suggestions
- **Live Timeline**: Real-time transcript and event tracking
- **Interactive Controls**: Accept/reject/edit AI recommendations

### Technical Highlights
- **Real-time WebSocket**: Live data streaming with Socket.IO
- **Conflict Detection Algorithm**: Haversine distance calculation with altitude separation
- **Responsive Design**: Modern UI with Tailwind CSS and radar-themed styling
- **Modular Architecture**: Clean separation between frontend and backend

## 🛠️ Tech Stack

### Frontend
- React 18
- D3.js for flight path visualization
- Tailwind CSS for styling
- Socket.IO client for real-time updates

### Backend
- Node.js with Express
- Socket.IO for WebSocket communication
- OpenSky Network API integration
- Cerebras API for AI inference
- Cron jobs for data fetching

## 🚀 Quick Start

### Prerequisites
- Node.js 16+ 
- npm or yarn

### Installation

1. **Clone and install dependencies:**
```bash
git clone <repository-url>
cd pennapps
npm run install-all
```

2. **Set up environment variables:**
```bash
cd server
cp env.example .env
# Edit .env with your Cerebras API key (optional for demo)
```

3. **Start the application:**
```bash
npm run dev
```

This will start both the backend server (port 5000) and React frontend (port 3000).

### Demo Mode
The application works out-of-the-box with mock data. For full functionality:
- Add your Cerebras API key to `server/.env`
- The system will automatically fetch live flight data from OpenSky

## 🎯 Demo Flow

1. **Start Live Feed**: Dashboard automatically connects to OpenSky API
2. **View Radar Map**: See flights around O'Hare (ORD) area
3. **Monitor Conflicts**: AI detects potential conflicts in real-time
4. **Review Recommendations**: Click on conflict alerts to see AI suggestions
5. **Take Action**: Accept, reject, or edit AI recommendations
6. **Timeline View**: Monitor all events in the live timeline panel

## 🔧 Configuration

### OpenSky Network
- No API key required for basic usage
- Rate limit: 10 requests per minute
- Data updates every 5 seconds

### Cerebras API (Optional)
- Add `CEREBRAS_API_KEY` to `server/.env`
- Falls back to mock recommendations if not configured

### Customization
- Modify `server/index.js` for different airports
- Update `client/src/components/RadarMap.js` for map projection
- Adjust conflict detection parameters in `server/index.js`

## 📊 API Endpoints

- `GET /api/flights` - Current flight data
- `GET /api/history` - Historical flight data
- `POST /api/conflict/resolve` - Resolve conflicts
- WebSocket: Real-time flight updates

## 🎨 UI Features

### Radar Aesthetic
- Dark background with neon-green flight paths
- Animated radar sweep effect
- Distance rings and compass rose
- Conflict zone highlighting

### Interactive Elements
- Click flights for detailed information
- Hover effects with flight data
- Real-time conflict alerts
- Timeline with color-coded events

## 🔮 Future Enhancements

- LiveATC audio integration with Whisper
- Incident playback mode
- Advanced AI training on historical data
- Multi-airport support
- Mobile-responsive design
- Export functionality for reports

## 🤝 Contributing

This is a PennApps hackathon project. Feel free to fork and extend!

## 📄 License

MIT License - see LICENSE file for details

---

**Built with ❤️ for PennApps 2024**
Repository For Penn Apps Hackathon 2025
