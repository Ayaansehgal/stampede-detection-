const mongoose = require('mongoose');

const emergencyLogSchema = new mongoose.Schema({
    eventId: { type: mongoose.Schema.Types.ObjectId, ref: 'Event', required: true },
    type: {
        type: String,
        enum: ['critical_risk', 'missing_person', 'emergency_escalation', 'barricade_control', 'evacuation'],
        required: true
    },
    zoneId: { type: String, default: 'all' },
    description: { type: String, default: '' },
    actionTaken: { type: String, default: '' },
    hash: { type: String, required: true },
    txHash: { type: String, default: '' },
    severity: { type: String, enum: ['low', 'medium', 'high', 'critical'], default: 'high' },
    resolved: { type: Boolean, default: false },
    timestamp: { type: Date, default: Date.now }
});

emergencyLogSchema.index({ eventId: 1, timestamp: -1 });

module.exports = mongoose.model('EmergencyLog', emergencyLogSchema);
