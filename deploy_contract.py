import json
import os
from web3 import Web3
from solcx import compile_standard, install_solc

def deploy_contract():

    print("Installing Solidity compiler (0.8.0)...")
    install_solc("0.8.0")

    with open("blockchain_contract/StampedeLogger.sol", "r") as f:
        contract_source = f.read()

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

    rpc_url = os.getenv("RPC_URL", "http://127.0.0.1:8545")
    print(f"Connecting to: {rpc_url}")
    w3 = Web3(Web3.HTTPProvider(rpc_url))

    if not w3.is_connected():
        print(f"Error: Could not connect to blockchain at {rpc_url}")
        print("Check your RPC_URL in .env or make sure Ganache is running.")
        return

    chain_id = int(os.getenv("CHAIN_ID", w3.eth.chain_id))
    print(f"Chain ID: {chain_id}")

    my_address = None
    private_key = os.getenv("PRIVATE_KEY")

    if len(w3.eth.accounts) > 0 and "127.0.0.1" in rpc_url:
        my_address = w3.eth.accounts[0]
        print(f"Deploying from Local Account: {my_address}")

        StampedeLogger = w3.eth.contract(abi=abi, bytecode=bytecode)
        tx_hash = StampedeLogger.constructor().transact({'from': my_address})
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    elif private_key:
        if private_key.startswith("YOUR_"):
             print("Error: Please update .env with your real PRIVATE_KEY")
             return

        account = w3.eth.account.from_key(private_key)
        my_address = account.address
        print(f"Deploying from Account: {my_address}")

        StampedeLogger = w3.eth.contract(abi=abi, bytecode=bytecode)
        nonce = w3.eth.get_transaction_count(my_address)

        tx = StampedeLogger.constructor().build_transaction({
            'chainId': chain_id,
            'gasPrice': w3.eth.gas_price,
            'from': my_address,
            'nonce': nonce
        })
        signed_tx = w3.eth.account.sign_transaction(tx, private_key)
        print("Sending transaction...")
        tx_hash = w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        print("Waiting for receipt...")
        tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)

    else:
        print("Error: No Local Accounts found and PRIVATE_KEY not set.")
        print("Please create a .env file with RPC_URL and PRIVATE_KEY.")
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

    contract_data = {
        "address": tx_receipt.contractAddress,
        "abi": abi
    }

    with open("server/contract_config.json", "w") as f:
        json.dump(contract_data, f, indent=4)

    print("Contract configuration saved to 'server/contract_config.json'")

if __name__ == "__main__":
    deploy_contract()
