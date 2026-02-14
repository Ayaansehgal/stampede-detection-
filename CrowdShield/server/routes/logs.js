const express = require('express');
const router = express.Router();
const EmergencyLog = require('../models/EmergencyLog');
const { getLedger } = require('../utils/blockchain');

// GET /api/logs/:eventId – Get all audit logs (emergency + blockchain)
router.get('/:eventId', async (req, res) => {
    try {
        const emergencyLogs = await EmergencyLog.find({ eventId: req.params.eventId }).sort({ timestamp: -1 });
        const blockchainRecords = getLedger(req.params.eventId);

        res.json({
            emergencyLogs,
            blockchainRecords,
            totalLogs: emergencyLogs.length,
            totalBlockchainEntries: blockchainRecords.length
        });
    } catch (err) {
        res.status(500).json({ error: err.message });
    }
});

module.exports = router;
