import os
import logging
from substrateinterface import Keypair

SEED_PATH = "wallets/mnemonic.txt"


def generate_mnemonic():
    if not os.path.exists(SEED_PATH):
        logging.info(f"No seed exists; creating")
        wallet = Keypair.generate_mnemonic()

        with open(SEED_PATH, "w") as f:
            f.write(wallet)
    else:
        logging.info(f"A seed already exists; skipping")


def get_mnemonic():
    if not os.path.exists(SEED_PATH):
        generate_mnemonic()

    with open(SEED_PATH, "r") as file:
        mnemonic = file.read()

    return mnemonic


def get_start_wallet(ss58: int):
    mnemonic = get_mnemonic()
    kp = Keypair.create_from_uri(f"{mnemonic}//startwallet", ss58_format=ss58)
    logging.info(f"Start Wallet: {kp.ss58_address}")
    return kp


def get_spam_wallets(ss58: int, amount: int):
    logging.info(f"Spam wallets: {amount}")
    mnemonic = get_mnemonic()
    kp_list = []
    pk_list = {}
    for i in range(amount):
        kp_list.append(
            Keypair.create_from_uri(f"{mnemonic}//spammening//{i}", ss58_format=ss58)
        )

    for key in kp_list:
        pk_list[key.ss58_address] = {
            "balance": None,
            "nonce": None,
        }
    return kp_list, pk_list


def split_list(lst, rpc_count):
    chunk_size = -(-len(lst) // rpc_count)
    return [lst[i : i + chunk_size] for i in range(0, len(lst), chunk_size)]


def count_funded_spam_acc(balance_list: list, wallet=None):
    cc = 0
    for acc, item in balance_list.items():
        if wallet:
            if acc == wallet.ss58_address:
                continue

        if item["balance"]:
            cc += 1
    return cc
