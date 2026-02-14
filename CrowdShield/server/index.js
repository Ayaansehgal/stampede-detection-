require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const http = require('http');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);

// Socket.io setup
const io = new Server(server, {
    cors: {
        origin: ['http://localhost:5173', 'http://localhost:3000'],
        methods: ['GET', 'POST']
    }
});
app.set('io', io);

// Middleware
app.use(cors());
app.use(express.json({ limit: '10mb' }));

// MongoDB Atlas Connection
const MONGODB_URI = process.env.MONGODB_URI || 'mongodb://localhost:27017/safewave';
mongoose.connect(MONGODB_URI)
    .then(() => console.log('✅ Connected to MongoDB Atlas'))
    .catch(err => console.error('❌ MongoDB connection error:', err.message));

// Routes
app.use('/api/events', require('./routes/events'));
app.use('/api/risk', require('./routes/risk'));
app.use('/api/emergency', require('./routes/emergency'));
app.use('/api/logs', require('./routes/logs'));
app.use('/api/compliance', require('./routes/compliance'));
app.use('/api/gates', require('./routes/gates'));
app.use('/api/pa', require('./routes/pa'));

// Health check
app.get('/api/health', (req, res) => {
    res.json({
        status: 'ok',
        service: 'SafeWave API',
        mongodb: mongoose.connection.readyState === 1 ? 'connected' : 'disconnected',
        uptime: process.uptime()
    });
});

// Socket.io connection handling
io.on('connection', (socket) => {
    console.log(`🔌 Client connected: ${socket.id}`);

    socket.on('joinEvent', (eventId) => {
        socket.join(eventId);
        console.log(`📡 Socket ${socket.id} joined event room: ${eventId}`);
    });

    socket.on('leaveEvent', (eventId) => {
        socket.leave(eventId);
        console.log(`📡 Socket ${socket.id} left event room: ${eventId}`);
    });

    socket.on('disconnect', () => {
        console.log(`🔌 Client disconnected: ${socket.id}`);
    });
});

// Start server
const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
    console.log(`\n🛡️  SafeWave API running on http://localhost:${PORT}`);
    console.log(`📡 Socket.io ready for connections`);
    console.log(`💾 MongoDB: ${MONGODB_URI.replace(/\/\/.*@/, '//<credentials>@')}\n`);
});
