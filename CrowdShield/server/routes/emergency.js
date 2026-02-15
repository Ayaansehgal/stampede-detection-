const express = require('express');
const router = express.Router();
const EmergencyLog = require('../models/EmergencyLog');
const GateState = require('../models/GateState');
const PABroadcast = require('../models/PABroadcast');
const Event = require('../models/Event');
const { logToBlockchain } = require('../utils/blockchain');

// Helper: Auto-lock ~50% of turnstiles across all gates
async function autoLockdownGates(eventId, reason, io) {
    const event = await Event.findById(eventId);
    if (!event) return;

    const totalGates = event.gates || 6;

    // Ensure gate records exist
    const existing = await GateState.find({ eventId });
    if (existing.length === 0) {
        const gateNames = ['Main Entry', 'North Gate', 'East Gate', 'West Gate', 'South Gate', 'VIP Entry'];
        const gates = [];
        for (let i = 1; i <= totalGates; i++) {
            gates.push({
                eventId,
                gateNumber: i,
                gateName: gateNames[i - 1] || `Gate ${i}`,
                turnstiles: Array(8).fill(true),
            });
        }
        await GateState.insertMany(gates);
    }

    // Lock every other turnstile (0, 2, 4, 6) across all gates
    const gates = await GateState.find({ eventId });
    for (const gate of gates) {
        gate.turnstiles = gate.turnstiles.map((_, i) => i % 2 !== 0);
        gate.updatedAt = new Date();
        gate.markModified('turnstiles');
        await gate.save();
    }

    // Broadcast gate update
    if (io) {
        const allGates = await GateState.find({ eventId }).sort({ gateNumber: 1 });
        io.to(eventId.toString()).emit('gateUpdate', allGates);
    }

    console.log(`[GATE-AUTO] Emergency lockdown: 50% turnstiles locked across all gates for event ${eventId}`);
    return gates.length;
}

// Helper: Auto-trigger PA broadcast
async function autoBroadcastPA(eventId, type, io) {
    const templates = {
        critical_risk: 'EMERGENCY: Crowd surge detected. All gates restricting entry. Please remain calm and follow security instructions. Do not push.',
        evacuation: 'EMERGENCY EVACUATION: All personnel proceed to nearest exit immediately. Follow illuminated exit signs. Do not use elevators.',
        stampede_alert: 'CRITICAL ALERT: Stampede risk detected. Entry gates are being closed. All security to positions. Crowd to remain stationary.',
    };

    const message = templates[type] || templates.critical_risk;

    const blockchainResult = await logToBlockchain({
        eventId, action: 'auto_pa_broadcast', type, message,
    });

    const broadcast = new PABroadcast({
        eventId,
        type: type === 'critical_risk' ? 'stampede_alert' : 'evacuation',
        message,
        zones: ['all'],
        triggeredBy: 'auto',
        priority: 'critical',
        status: 'active',
        hash: blockchainResult.hash,
    });

    await broadcast.save();

    if (io) {
        io.to(eventId.toString()).emit('paBroadcast', {
            _id: broadcast._id,
            type: broadcast.type,
            message,
            zones: ['all'],
            priority: 'critical',
            triggeredBy: 'auto',
            status: 'active',
            hash: blockchainResult.hash,
            timestamp: broadcast.timestamp,
        });
    }

    console.log(`[PA-AUTO] Critical broadcast triggered for event ${eventId}: "${message.substring(0, 60)}..."`);
    return broadcast;
}

// POST /api/emergency – Trigger emergency event
router.post('/', async (req, res) => {
    try {
        const { eventId, type, zoneId, description } = req.body;

        if (!eventId || !type) {
            return res.status(400).json({ error: 'Missing required fields: eventId, type' });
        }

        // Generate action based on type
        let actionTaken = '';
        let severity = 'high';
        switch (type) {
            case 'critical_risk':
                actionTaken = 'Initiated emergency protocol. Entry gates auto-closed. PA system broadcasting stampede alert.';
                severity = 'critical';
                break;
            case 'missing_person':
                actionTaken = 'Missing person alert broadcast to all zones. PA announcement triggered. All cameras on sweep mode.';
                severity = 'high';
                break;
            case 'emergency_escalation':
                actionTaken = 'Emergency escalated to local authorities. Medical team dispatched. Zone perimeter locked.';
                severity = 'critical';
                break;
            case 'barricade_control':
                actionTaken = 'Smart barricades activated. ESP32 actuator signal sent. Traffic rerouted.';
                severity = 'medium';
                break;
            case 'evacuation':
                actionTaken = 'Full venue evacuation initiated. All entry gates closed. PA announcing exit routes.';
                severity = 'critical';
                break;
            default:
                actionTaken = 'Emergency event logged.';
        }

        // Log to blockchain
        const blockchainResult = await logToBlockchain({
            eventId, type, zoneId: zoneId || 'all',
            description: description || '', actionTaken, severity,
        });

        const log = new EmergencyLog({
            eventId, type, zoneId: zoneId || 'all',
            description: description || '', actionTaken,
            hash: blockchainResult.hash, txHash: blockchainResult.txHash, severity,
        });

        await log.save();

        // Get Socket.io instance
        const io = req.app.get('io');

        // Broadcast emergency to all connected clients
        if (io) {
            io.to(eventId.toString()).emit('emergency', {
                type, zoneId: zoneId || 'all',
                description: description || '', actionTaken, severity,
                hash: blockchainResult.hash, txHash: blockchainResult.txHash,
                timestamp: log.timestamp,
            });
        }

        // === AUTO-TRIGGER: Gate lockdown + PA broadcast ===
        let autoGates = null;
        let autoPa = null;

        if (type === 'critical_risk' || type === 'evacuation') {
            // Auto-close ~50% of entry gates
            autoGates = await autoLockdownGates(eventId, `Auto: ${type} in zone ${zoneId || 'all'}`, io);
            // Auto-trigger PA broadcast
            autoPa = await autoBroadcastPA(eventId, type, io);
        }

        if (type === 'missing_person') {
            // Auto-trigger missing person PA (but don't close gates)
            autoPa = await autoBroadcastPA(eventId, 'missing_person', io);
        }

        // ESP32 actuator signal
        if (type === 'barricade_control' || type === 'evacuation' || type === 'critical_risk') {
            console.log(`[ACTUATOR] ESP32 signal sent: ${type} -> Zone ${zoneId || 'ALL'}`);
        }

        res.status(201).json({
            log,
            autoActions: {
                gatesClosed: autoGates,
                paBroadcast: autoPa ? { _id: autoPa._id, message: autoPa.message } : null,
            },
        });
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// GET /api/emergency/:eventId – Get emergency logs for event
router.get('/:eventId', async (req, res) => {
    try {
        const logs = await EmergencyLog.find({ eventId: req.params.eventId }).sort({ timestamp: -1 });
        res.json(logs);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
