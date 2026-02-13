
def generate_hash(data):
    import hashlib
    return hashlib.sha256(str(data).encode()).hexdigest()

def log_to_blockchain(event_data):
    print(f"\n[MOCK BLOCKCHAIN] LOG ENTRY:")
    print(f"  Timestamp: {event_data['timestamp']}")
    print(f"  Label:     {event_data['label']}")
    print(f"  Score:     {event_data['risk_score']:.4f}")
    print(f"  Hash:      {event_data['hash']}")
    print("--------------------------------------\n")