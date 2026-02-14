const mongoose = require('mongoose');

const eventSchema = new mongoose.Schema({
    name: { type: String, required: true },
    type: { type: String, enum: ['concert', 'temple', 'rally', 'sports', 'festival', 'other'], default: 'other' },
    expectedCrowd: { type: Number, required: true },
    venueArea: { type: Number, required: true },
    gates: { type: Number, default: 2 },
    mapUrl: { type: String, default: '' },
    zones: [{
        id: String,
        name: String,
        capacity: Number,
        type: { type: String, enum: ['entry', 'exit', 'stage', 'open', 'vip', 'food'], default: 'open' }
    }],
    status: { type: String, enum: ['planning', 'live', 'completed'], default: 'planning' },
    safetyScore: { type: Number, default: 0 },
}, { timestamps: true });

module.exports = mongoose.model('Event', eventSchema);
