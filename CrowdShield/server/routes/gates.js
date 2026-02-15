const express = require('express');
const router = express.Router();
const GateState = require('../models/GateState');
const Event = require('../models/Event');
const { logToBlockchain } = require('../utils/blockchain');

const TURNSTILES_PER_GATE = 8;

// Helper: Initialize gates for event if they don't exist
async function ensureGates(eventId) {
    const event = await Event.findById(eventId);
    if (!event) throw new Error('Event not found');

    const existing = await GateState.find({ eventId });
    if (existing.length === 0) {
        const gateNames = ['Main Entry', 'North Gate', 'East Gate', 'West Gate', 'South Gate', 'VIP Entry'];
        const gates = [];
        for (let i = 1; i <= (event.gates || 6); i++) {
            gates.push({
                eventId,
                gateNumber: i,
                gateName: gateNames[i - 1] || `Gate ${i}`,
                turnstiles: Array(TURNSTILES_PER_GATE).fill(true),  // all unlocked
            });
        }
        await GateState.insertMany(gates);
    }
    return event;
}

// GET /api/gates/:eventId — Get all gate states
router.get('/:eventId', async (req, res) => {
    try {
        await ensureGates(req.params.eventId);
        const gates = await GateState.find({ eventId: req.params.eventId }).sort({ gateNumber: 1 });
        res.json(gates);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// POST /api/gates/toggle-turnstile — Lock/unlock a single turnstile
router.post('/toggle-turnstile', async (req, res) => {
    try {
        const { eventId, gateNumber, turnstileIndex } = req.body;
        if (!eventId || !gateNumber || turnstileIndex == null) {
            return res.status(400).json({ error: 'Missing: eventId, gateNumber, turnstileIndex' });
        }

        await ensureGates(eventId);

        const gate = await GateState.findOne({ eventId, gateNumber });
        if (!gate) return res.status(404).json({ error: `Gate ${gateNumber} not found` });

        // Toggle the specific turnstile
        gate.turnstiles[turnstileIndex] = !gate.turnstiles[turnstileIndex];
        gate.updatedAt = new Date();
        gate.markModified('turnstiles');
        await gate.save();

        // Broadcast via Socket.io
        const io = req.app.get('io');
        if (io) {
            const allGates = await GateState.find({ eventId }).sort({ gateNumber: 1 });
            io.to(eventId.toString()).emit('gateUpdate', allGates);
        }

        const action = gate.turnstiles[turnstileIndex] ? 'unlocked' : 'locked';
        console.log(`[GATE] Gate ${gateNumber} turnstile ${turnstileIndex + 1} → ${action}`);
        res.json(gate);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// POST /api/gates/lock-turnstiles — Lock multiple turnstiles on a gate
router.post('/lock-turnstiles', async (req, res) => {
    try {
        const { eventId, gateNumber, turnstileIndices } = req.body;
        if (!eventId || !gateNumber || !turnstileIndices) {
            return res.status(400).json({ error: 'Missing: eventId, gateNumber, turnstileIndices' });
        }

        await ensureGates(eventId);

        const gate = await GateState.findOne({ eventId, gateNumber });
        if (!gate) return res.status(404).json({ error: `Gate ${gateNumber} not found` });

        turnstileIndices.forEach(i => { gate.turnstiles[i] = false; });
        gate.updatedAt = new Date();
        gate.markModified('turnstiles');
        await gate.save();

        const io = req.app.get('io');
        if (io) {
            const allGates = await GateState.find({ eventId }).sort({ gateNumber: 1 });
            io.to(eventId.toString()).emit('gateUpdate', allGates);
        }

        res.json(gate);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// POST /api/gates/auto-lockdown — Auto-lock ~50% turnstiles across all gates
router.post('/auto-lockdown', async (req, res) => {
    try {
        const { eventId, reason } = req.body;
        if (!eventId) return res.status(400).json({ error: 'Missing eventId' });

        await ensureGates(eventId);

        const gates = await GateState.find({ eventId });
        for (const gate of gates) {
            // Lock every other turnstile (0, 2, 4, 6)
            gate.turnstiles = gate.turnstiles.map((_, i) => i % 2 !== 0);
            gate.updatedAt = new Date();
            gate.markModified('turnstiles');
            await gate.save();
        }

        await logToBlockchain({
            eventId, action: 'auto_gate_lockdown', reason: reason || 'Stampede response',
            turnstilesLocked: '50% across all gates',
        });

        const allGates = await GateState.find({ eventId }).sort({ gateNumber: 1 });
        const io = req.app.get('io');
        if (io) io.to(eventId.toString()).emit('gateUpdate', allGates);

        console.log(`[GATE-AUTO] Locked 50% turnstiles across all gates — ${reason}`);
        res.json(allGates);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// POST /api/gates/open-all — Unlock all turnstiles
router.post('/open-all', async (req, res) => {
    try {
        const { eventId } = req.body;
        if (!eventId) return res.status(400).json({ error: 'Missing eventId' });

        const gates = await GateState.find({ eventId });
        for (const gate of gates) {
            gate.turnstiles = gate.turnstiles.map(() => true);
            gate.updatedAt = new Date();
            gate.markModified('turnstiles');
            await gate.save();
        }

        const allGates = await GateState.find({ eventId }).sort({ gateNumber: 1 });
        const io = req.app.get('io');
        if (io) io.to(eventId.toString()).emit('gateUpdate', allGates);

        console.log(`[GATE] All turnstiles unlocked for event ${eventId}`);
        res.json(allGates);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

module.exports = router;
