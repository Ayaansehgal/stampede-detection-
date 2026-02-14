const mongoose = require('mongoose');

const complianceReportSchema = new mongoose.Schema({
    eventId: { type: mongoose.Schema.Types.ObjectId, ref: 'Event', required: true },
    safetyScore: { type: Number, default: 0, min: 0, max: 100 },
    findings: [{
        category: String,
        status: { type: String, enum: ['pass', 'warning', 'fail'] },
        message: String,
        score: Number
    }],
    blockchainHashes: [String],
    totalReadings: { type: Number, default: 0 },
    totalEmergencies: { type: Number, default: 0 },
    avgRiskScore: { type: Number, default: 0 },
    generatedAt: { type: Date, default: Date.now }
});

module.exports = mongoose.model('ComplianceReport', complianceReportSchema);
