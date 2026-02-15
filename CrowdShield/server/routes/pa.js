const express = require('express');
const router = express.Router();
const PABroadcast = require('../models/PABroadcast');
const { logToBlockchain } = require('../utils/blockchain');

// Pre-defined broadcast templates
const BROADCAST_TEMPLATES = {
    evacuation: 'ATTENTION: Immediate evacuation required. Please proceed to the nearest exit calmly. Follow the directions of security personnel.',
    stampede_alert: 'ATTENTION: Crowd surge detected. All personnel maintain positions. Entry gates are being restricted. Do not push. Stay calm and follow instructions.',
    missing_person: 'ATTENTION: A missing person alert has been issued. Please check your surroundings and report to the nearest security checkpoint if you see the described individual.',
};

// GET /api/pa/:eventId — Get PA broadcast history
router.get('/:eventId', async (req, res) => {
    try {
        const broadcasts = await PABroadcast.find({ eventId: req.params.eventId })
            .sort({ timestamp: -1 })
            .limit(50);
        res.json(broadcasts);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/pa/:eventId/active — Get currently active broadcasts
router.get('/:eventId/active', async (req, res) => {
    try {
        const active = await PABroadcast.find({ eventId: req.params.eventId, status: 'active' })
            .sort({ timestamp: -1 });
        res.json(active);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// POST /api/pa/broadcast — Trigger a PA override broadcast
router.post('/broadcast', async (req, res) => {
    try {
        const { eventId, type, message, zones, priority, triggeredBy } = req.body;

        if (!eventId || !type) {
            return res.status(400).json({ error: 'Missing required fields: eventId, type' });
        }

        // Use template message if none provided
        const broadcastMessage = message || BROADCAST_TEMPLATES[type] || 'Emergency announcement in progress.';
        const targetZones = zones && zones.length > 0 ? zones : ['all'];

        // Log to blockchain
        const blockchainResult = await logToBlockchain({
            eventId,
            action: 'pa_broadcast',
            type,
            message: broadcastMessage,
            zones: targetZones,
            priority: priority || 'urgent',
        });

        const broadcast = new PABroadcast({
            eventId,
            type,
            message: broadcastMessage,
            zones: targetZones,
            triggeredBy: triggeredBy || 'manual',
            priority: priority || 'urgent',
            status: 'active',
            hash: blockchainResult.hash,
        });

        await broadcast.save();

        // Broadcast to all connected clients via Socket.io
        const io = req.app.get('io');
        if (io) {
            io.to(eventId.toString()).emit('paBroadcast', {
                _id: broadcast._id,
                type,
                message: broadcastMessage,
                zones: targetZones,
                priority: priority || 'urgent',
                status: 'active',
                hash: blockchainResult.hash,
                timestamp: broadcast.timestamp,
            });
        }

        console.log(`[PA] ${priority || 'urgent'} broadcast: "${broadcastMessage.substring(0, 60)}..." → Zones: ${targetZones.join(', ')}`);
        res.status(201).json(broadcast);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// POST /api/pa/stop — Stop an active broadcast
router.post('/stop', async (req, res) => {
    try {
        const { eventId, broadcastId } = req.body;
        if (!eventId) return res.status(400).json({ error: 'Missing eventId' });

        if (broadcastId) {
            // Stop specific broadcast
            await PABroadcast.findByIdAndUpdate(broadcastId, { status: 'completed' });
        } else {
            // Stop all active broadcasts for event
            await PABroadcast.updateMany(
                { eventId, status: 'active' },
                { status: 'completed' }
            );
        }

        // Notify clients
        const io = req.app.get('io');
        if (io) {
            io.to(eventId.toString()).emit('paStop', { broadcastId: broadcastId || 'all' });
        }

        console.log(`[PA] Broadcast stopped: ${broadcastId || 'all'} for event ${eventId}`);
        res.json({ success: true });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

module.exports = router;
