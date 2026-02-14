const mongoose = require('mongoose');

const gateStateSchema = new mongoose.Schema({
    eventId: { type: mongoose.Schema.Types.ObjectId, ref: 'Event', required: true },
    gateNumber: { type: Number, required: true },
    gateName: { type: String, default: '' },
    // Array of turnstile states: true = unlocked (open), false = locked
    turnstiles: {
        type: [Boolean],
        default: () => Array(8).fill(true),  // 8 turnstiles per gate, all unlocked by default
    },
    updatedAt: { type: Date, default: Date.now },
});

gateStateSchema.index({ eventId: 1, gateNumber: 1 }, { unique: true });

// Virtual: how many turnstiles are open
gateStateSchema.virtual('openCount').get(function () {
    return this.turnstiles.filter(t => t).length;
});

gateStateSchema.virtual('lockedCount').get(function () {
    return this.turnstiles.filter(t => !t).length;
});

gateStateSchema.set('toJSON', { virtuals: true });

module.exports = mongoose.model('GateState', gateStateSchema);
