const mongoose = require('mongoose');

const zoneReadingSchema = new mongoose.Schema({
    eventId: { type: mongoose.Schema.Types.ObjectId, ref: 'Event', required: true },
    zoneId: { type: String, required: true },
    personCount: { type: Number, required: true },
    instabilityScore: { type: Number, required: true, min: 0, max: 1 },
    riskLevel: { type: String, enum: ['green', 'yellow', 'red'], default: 'green' },
    riskScore: { type: Number, default: 0 },
    recommendation: { type: String, default: '' },
    timestamp: { type: Date, default: Date.now }
});

zoneReadingSchema.index({ eventId: 1, timestamp: -1 });
zoneReadingSchema.index({ eventId: 1, zoneId: 1, timestamp: -1 });

module.exports = mongoose.model('ZoneReading', zoneReadingSchema);
