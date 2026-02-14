const express = require('express');
const router = express.Router();
const Event = require('../models/Event');

// POST /api/events – Create new event
router.post('/', async (req, res) => {
    try {
        const { name, type, expectedCrowd, venueArea, gates, mapUrl } = req.body;

        // Auto-generate default zones based on gates
        const zoneCount = Math.max(4, (gates || 2) * 2);
        const zoneLabels = [];
        for (let i = 0; i < zoneCount; i++) {
            const letter = String.fromCharCode(65 + Math.floor(i / 3));
            const num = (i % 3) + 1;
            zoneLabels.push({
                id: `${letter}${num}`,
                name: `Zone ${letter}${num}`,
                capacity: Math.round((expectedCrowd || 1000) / zoneCount),
                type: i < gates ? 'entry' : i < gates * 2 ? 'exit' : 'open'
            });
        }

        // Compute initial safety score
        const density = (expectedCrowd || 0) / (venueArea || 1);
        const gateRatio = (expectedCrowd || 1000) / (gates || 2);
        let safetyScore = 85;
        if (density > 4) safetyScore -= 30;
        else if (density > 2) safetyScore -= 15;
        if (gateRatio > 2000) safetyScore -= 20;
        else if (gateRatio > 1000) safetyScore -= 10;

        const event = new Event({
            name,
            type: type || 'other',
            expectedCrowd,
            venueArea,
            gates: gates || 2,
            mapUrl: mapUrl || '',
            zones: zoneLabels,
            safetyScore: Math.max(0, safetyScore)
        });

        await event.save();
        res.status(201).json(event);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// GET /api/events – Get all events
router.get('/', async (req, res) => {
    try {
        const events = await Event.find().sort({ createdAt: -1 });
        res.json(events);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/events/:id – Get single event
router.get('/:id', async (req, res) => {
    try {
        const event = await Event.findById(req.params.id);
        if (!event) return res.status(404).json({ error: 'Event not found' });
        res.json(event);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// PATCH /api/events/:id – Update event status
router.patch('/:id', async (req, res) => {
    try {
        const event = await Event.findByIdAndUpdate(req.params.id, req.body, { new: true });
        if (!event) return res.status(404).json({ error: 'Event not found' });
        res.json(event);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

module.exports = router;
