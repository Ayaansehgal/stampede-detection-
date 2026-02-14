import json
import os
from web3 import Web3
from solcx import compile_standard, install_solc

def deploy_contract():
    # 1. Install Solc (Solidity Compiler)
    print("Installing Solidity compiler (0.8.0)...")
    install_solc("0.8.0")

    # 2. Read Solidity Source
    with open("blockchain_contract/StampedeLogger.sol", "r") as f:
        contract_source = f.read()

    # 3. Compile
    print("Compiling contract...")
    compiled_sol = compile_standard(
        {
            "language": "Solidity",
            "sources": {"StampedeLogger.sol": {"content": contract_source}},
            "settings": {
                "outputSelection": {
                    "*": {
                        "*": ["abi", "metadata", "evm.bytecode", "evm.sourceMap"]
                    }
                }
            },
        },
        solc_version="0.8.0",
    )

    bytecode = compiled_sol["contracts"]["StampedeLogger.sol"]["StampedeLogger"]["evm"]["bytecode"]["object"]
    abi = compiled_sol["contracts"]["StampedeLogger.sol"]["StampedeLogger"]["abi"]

    # 4. Connect to Blockchain (Local Ganache by default)
    #    To use Sepolia/Goerli, change HTTPProvider to your Infura/Alchemy URL
    rpc_url = "http://127.0.0.1:8545" 
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    
    if not w3.is_connected():
        print(f"Error: Could not connect to blockchain at {rpc_url}")
        print("Make sure Ganache or a local node is running!")
        return

    chain_id = w3.eth.chain_id
    
    # 5. Setup Account (First account from Ganache usually)
    #    For real testnet, replace with your PRIVATE_KEY env variable
    if len(w3.eth.accounts) > 0:
        my_address = w3.eth.accounts[0]
        print(f"Deploying from address: {my_address}")
        
        # Simple deployment (unlocked account)
        StampedeLogger = w3.eth.contract(abi=abi, bytecode=bytecode)
        tx_hash = StampedeLogger.constructor().transact({'from': my_address})
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    else:
        # If no accounts (e.g. Infura), need private key
        private_key = os.getenv("PRIVATE_KEY", "YOUR_PRIVATE_KEY_HERE")
        if "YOUR_PRIVATE_KEY" in private_key:
             print("Error: No accounts found and no PRIVATE_KEY set.")
             return
        
        account = w3.eth.account.from_key(private_key)
        print(f"Deploying from address: {account.address}")
        
        StampedeLogger = w3.eth.contract(abi=abi, bytecode=bytecode)
        nonce = w3.eth.get_transaction_count(account.address)
        
        tx = StampedeLogger.constructor().build_transaction({
            'chainId': chain_id,
            'gasPrice': w3.eth.gas_price,
            'from': account.address,
            'nonce': nonce
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    print(f"Contract deployed to: {tx_receipt.contractAddress}")

    # 6. Save Contract Data for Application
    contract_data = {
        "address": tx_receipt.contractAddress,
        "abi": abi
    }
    
    with open("server/contract_config.json", "w") as f:
        json.dump(contract_data, f, indent=4)
    
    print("Contract configuration saved to 'server/contract_config.json'")

if __name__ == "__main__":
    deploy_contract()
