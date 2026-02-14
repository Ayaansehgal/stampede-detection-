const crypto = require('crypto');

// In-memory blockchain ledger (simulated)
const blockchainLedger = [];

/**
 * Generate SHA256 hash from event data
 */
function generateHash(data) {
    const str = JSON.stringify(data) + Date.now().toString();
    return crypto.createHash('sha256').update(str).digest('hex');
}

/**
 * Log event to simulated blockchain
 * Returns { hash, txHash, timestamp, blockNumber }
 */
async function logToBlockchain(eventData) {
    const hash = generateHash(eventData);
    const txHash = '0x' + crypto.randomBytes(32).toString('hex');
    const blockNumber = blockchainLedger.length + 1;
    const timestamp = new Date().toISOString();

    const record = {
        hash,
        txHash,
        blockNumber,
        data: eventData,
        timestamp,
        verified: true
    };

    blockchainLedger.push(record);

    // If real blockchain env vars are set, attempt Ethers.js call
    if (process.env.BLOCKCHAIN_RPC_URL && process.env.BLOCKCHAIN_PRIVATE_KEY && process.env.BLOCKCHAIN_CONTRACT_ADDRESS) {
        try {
            const { ethers } = require('ethers');
            const provider = new ethers.JsonRpcProvider(process.env.BLOCKCHAIN_RPC_URL);
            const wallet = new ethers.Wallet(process.env.BLOCKCHAIN_PRIVATE_KEY, provider);
            const abi = ['function logEvent(string memory eventHash) public'];
            const contract = new ethers.Contract(process.env.BLOCKCHAIN_CONTRACT_ADDRESS, abi, wallet);
            const tx = await contract.logEvent(hash);
            await tx.wait();
            record.txHash = tx.hash;
            record.onChain = true;
        } catch (err) {
            console.warn('Blockchain call failed, using simulated log:', err.message);
            record.onChain = false;
        }
    }

    return { hash, txHash: record.txHash, timestamp, blockNumber };
}

/**
 * Get all blockchain records for an event
 */
function getLedger(eventId) {
    if (!eventId) return blockchainLedger;
    return blockchainLedger.filter(r => r.data?.eventId?.toString() === eventId.toString());
}

module.exports = { generateHash, logToBlockchain, getLedger };
