const express = require('express');
const router = express.Router();
const Event = require('../models/Event');
const ZoneReading = require('../models/ZoneReading');
const EmergencyLog = require('../models/EmergencyLog');
const ComplianceReport = require('../models/ComplianceReport');
const { computeSafetyScore, generateFindings } = require('../utils/riskEngine');
const { getLedger } = require('../utils/blockchain');

// GET /api/compliance/:eventId – Generate/fetch compliance report
router.get('/:eventId', async (req, res) => {
    try {
        const event = await Event.findById(req.params.eventId);
        if (!event) return res.status(404).json({ error: 'Event not found' });

        const readings = await ZoneReading.find({ eventId: req.params.eventId });
        const emergencies = await EmergencyLog.find({ eventId: req.params.eventId });
        const blockchainRecords = getLedger(req.params.eventId);

        const findings = generateFindings(event, readings, emergencies.length);
        const safetyScore = computeSafetyScore(readings, emergencies.length);

        // Upsert compliance report
        const report = await ComplianceReport.findOneAndUpdate(
            { eventId: req.params.eventId },
            {
                safetyScore,
                findings,
                blockchainHashes: blockchainRecords.map(r => r.hash),
                totalReadings: readings.length,
                totalEmergencies: emergencies.length,
                avgRiskScore: readings.length > 0
                    ? Math.round((readings.reduce((s, r) => s + (r.riskScore || 0), 0) / readings.length) * 100) / 100
                    : 0,
                generatedAt: new Date()
            },
            { new: true, upsert: true }
        );

        res.json(report);
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
