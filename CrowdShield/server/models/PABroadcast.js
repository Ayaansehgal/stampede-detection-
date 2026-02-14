const mongoose = require('mongoose');

const paBroadcastSchema = new mongoose.Schema({
    eventId: { type: mongoose.Schema.Types.ObjectId, ref: 'Event', required: true },
    type: {
        type: String,
        enum: ['evacuation', 'missing_person', 'stampede_alert', 'custom'],
        required: true,
    },
    message: { type: String, required: true },
    zones: [{ type: String }],               // ['A1','B1'] or ['all']
    triggeredBy: { type: String, enum: ['auto', 'manual'], default: 'manual' },
    priority: { type: String, enum: ['normal', 'urgent', 'critical'], default: 'urgent' },
    status: { type: String, enum: ['active', 'completed'], default: 'active' },
    hash: { type: String, default: '' },
    timestamp: { type: Date, default: Date.now },
});

paBroadcastSchema.index({ eventId: 1, timestamp: -1 });

module.exports = mongoose.model('PABroadcast', paBroadcastSchema);
