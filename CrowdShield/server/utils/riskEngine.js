/**
 * Risk Engine – Rule-based risk computation
 * 
 * riskScore = instabilityScore * 0.6 + (personCount / capacity) * 0.4
 * Green: < 0.4  |  Yellow: 0.4–0.7  |  Red: > 0.7
 */

const DEFAULT_ZONE_CAPACITY = 200;

function computeRisk(personCount, instabilityScore, zoneCapacity = DEFAULT_ZONE_CAPACITY) {
    const occupancyRatio = Math.min(personCount / zoneCapacity, 1.5);
    const riskScore = (instabilityScore * 0.6) + (occupancyRatio * 0.4);

    let riskLevel;
    if (riskScore < 0.4) riskLevel = 'green';
    else if (riskScore <= 0.7) riskLevel = 'yellow';
    else riskLevel = 'red';

    return { riskScore: Math.round(riskScore * 100) / 100, riskLevel };
}

function getRecommendation(riskLevel, personCount, instabilityScore, zoneId) {
    if (riskLevel === 'red' && instabilityScore > 0.7) {
        return `⚠️ CRITICAL: Zone ${zoneId} – Evacuate immediately. High instability (${instabilityScore}) with ${personCount} people.`;
    }
    if (riskLevel === 'red') {
        return `🚨 Zone ${zoneId} – Open additional exit gates. Crowd count (${personCount}) exceeds safe threshold.`;
    }
    if (riskLevel === 'yellow' && instabilityScore > 0.5) {
        return `⚠️ Zone ${zoneId} – Slow entry rate. Rising instability detected (${instabilityScore}).`;
    }
    if (riskLevel === 'yellow') {
        return `🟡 Zone ${zoneId} – Monitor closely. Consider redirecting flow to adjacent zones.`;
    }
    return `✅ Zone ${zoneId} – Normal operations. Crowd density within safe limits.`;
}

/**
 * Compute safety score for an event (0–100)
 */
function computeSafetyScore(readings, emergencies) {
    if (!readings || readings.length === 0) return 85; // default for new events

    const avgRisk = readings.reduce((sum, r) => sum + (r.riskScore || 0), 0) / readings.length;
    const redCount = readings.filter(r => r.riskLevel === 'red').length;
    const emergencyPenalty = Math.min((emergencies || 0) * 5, 30);
    const redPenalty = Math.min(redCount * 3, 25);

    let score = 100 - (avgRisk * 40) - redPenalty - emergencyPenalty;
    return Math.max(0, Math.min(100, Math.round(score)));
}

/**
 * Generate compliance findings
 */
function generateFindings(event, readings, emergencies) {
    const findings = [];
    const avgRisk = readings.length > 0
        ? readings.reduce((sum, r) => sum + (r.riskScore || 0), 0) / readings.length
        : 0;
    const redReadings = readings.filter(r => r.riskLevel === 'red').length;

    // Capacity check
    const totalPeople = readings.length > 0
        ? Math.max(...readings.map(r => r.personCount || 0))
        : 0;
    const capacityRatio = totalPeople / (event.expectedCrowd || 1);
    findings.push({
        category: 'Crowd Capacity',
        status: capacityRatio > 1.0 ? 'fail' : capacityRatio > 0.8 ? 'warning' : 'pass',
        message: capacityRatio > 1.0
            ? `Peak crowd (${totalPeople}) exceeded expected capacity (${event.expectedCrowd})`
            : `Peak crowd within expected capacity limits`,
        score: capacityRatio > 1.0 ? 30 : capacityRatio > 0.8 ? 70 : 95
    });

    // Risk level distribution
    findings.push({
        category: 'Risk Incidents',
        status: redReadings > 5 ? 'fail' : redReadings > 0 ? 'warning' : 'pass',
        message: redReadings > 0
            ? `${redReadings} critical (red) risk readings detected`
            : 'No critical risk readings during event',
        score: Math.max(0, 100 - redReadings * 10)
    });

    // Emergency response
    findings.push({
        category: 'Emergency Response',
        status: emergencies > 3 ? 'fail' : emergencies > 0 ? 'warning' : 'pass',
        message: emergencies > 0
            ? `${emergencies} emergency events triggered and logged`
            : 'No emergency events during the event',
        score: Math.max(0, 100 - emergencies * 15)
    });

    // Gate adequacy
    const gateRatio = (event.expectedCrowd || 1000) / (event.gates || 2);
    findings.push({
        category: 'Gate Adequacy',
        status: gateRatio > 2000 ? 'fail' : gateRatio > 1000 ? 'warning' : 'pass',
        message: gateRatio > 2000
            ? `Insufficient gates: ${gateRatio.toFixed(0)} people per gate (recommended < 1000)`
            : `Gate ratio acceptable: ${gateRatio.toFixed(0)} people per gate`,
        score: gateRatio > 2000 ? 40 : gateRatio > 1000 ? 70 : 95
    });

    // Venue density
    const density = (event.expectedCrowd || 0) / (event.venueArea || 1);
    findings.push({
        category: 'Venue Density',
        status: density > 4 ? 'fail' : density > 2 ? 'warning' : 'pass',
        message: `Crowd density: ${density.toFixed(2)} persons/sqm (safe limit: 2/sqm)`,
        score: density > 4 ? 25 : density > 2 ? 60 : 95
    });

    return findings;
}

module.exports = { computeRisk, getRecommendation, computeSafetyScore, generateFindings };
