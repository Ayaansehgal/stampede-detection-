const express = require('express');
const router = express.Router();
const ZoneReading = require('../models/ZoneReading');
const Event = require('../models/Event');
const { computeRisk, getRecommendation } = require('../utils/riskEngine');

// POST /api/risk – Ingest sensor data and compute risk
router.post('/', async (req, res) => {
    try {
        const { eventId, zoneId, personCount, instabilityScore } = req.body;

        if (!eventId || !zoneId || personCount == null || instabilityScore == null) {
            return res.status(400).json({ error: 'Missing required fields: eventId, zoneId, personCount, instabilityScore' });
        }

        // Get zone capacity from event
        const event = await Event.findById(eventId);
        let zoneCapacity = 200;
        if (event) {
            const zone = event.zones.find(z => z.id === zoneId);
            if (zone) zoneCapacity = zone.capacity || 200;
        }

        const { riskScore, riskLevel } = computeRisk(personCount, instabilityScore, zoneCapacity);
        const recommendation = getRecommendation(riskLevel, personCount, instabilityScore, zoneId);

        const reading = new ZoneReading({
            eventId,
            zoneId,
            personCount,
            instabilityScore,
            riskLevel,
            riskScore,
            recommendation
        });

        await reading.save();

        // Broadcast via Socket.io (attached in index.js)
        const io = req.app.get('io');
        if (io) {
            io.to(eventId).emit('riskUpdate', {
                zoneId,
                personCount,
                instabilityScore,
                riskLevel,
                riskScore,
                recommendation,
                timestamp: reading.timestamp
            });
        }

        res.status(201).json(reading);
    } catch (err) {
        res.status(400).json({ error: err.message });
    }
});

// GET /api/risk/:eventId – Get recent readings for an event
router.get('/:eventId', async (req, res) => {
    try {
        const { zoneId, minutes } = req.query;
        const filter = { eventId: req.params.eventId };

        if (zoneId) filter.zoneId = zoneId;
        if (minutes) {
            filter.timestamp = { $gte: new Date(Date.now() - parseInt(minutes) * 60 * 1000) };
        }

        const readings = await ZoneReading.find(filter).sort({ timestamp: -1 }).limit(200);
        res.json(readings);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

// GET /api/risk/:eventId/latest – Get latest reading per zone
router.get('/:eventId/latest', async (req, res) => {
    try {
        const readings = await ZoneReading.aggregate([
            { $match: { eventId: req.params.eventId } },
            { $sort: { timestamp: -1 } },
            {
                $group: {
                    _id: '$zoneId',
                    personCount: { $first: '$personCount' },
                    instabilityScore: { $first: '$instabilityScore' },
                    riskLevel: { $first: '$riskLevel' },
                    riskScore: { $first: '$riskScore' },
                    recommendation: { $first: '$recommendation' },
                    timestamp: { $first: '$timestamp' }
                }
            }
        ]);
        res.json(readings);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
