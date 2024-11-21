import copy
import json
import logging
import random
import time
import sys
import threading
from hashlib import blake2b
from unittest import skip
from scalecodec.base import ScaleBytes
from substrateinterface import SubstrateInterface, Keypair
from substrateinterface.storage import StorageKey
from websocket import WebSocketConnectionClosedException
from utils import get_start_wallet, get_spam_wallets, split_list, count_funded_spam_acc

BALANCES_LIST = {}
DATA_LOCK = threading.Lock()
NEW_BLOCK = 0


def update_balances(accounts: dict, rpcurl: str) -> None:
    global BALANCES_LIST
    logging.info("Start Thread to update balances")
    sbstr = SubstrateInterface(url=rpcurl)
    sbstr.init_runtime()

    _last_used_block = 0
    while True:
        try:
            tnow = time.time()
            storage_keys = [
                StorageKey.create_from_storage_function(
                    "System",
                    "Account",
                    [acc],
                    runtime_config=sbstr.runtime_config,
                    metadata=sbstr.metadata,
                )
                for acc, info in accounts.items()
            ]
            storagetime = time.time() - tnow

            tnow = time.time()
            results = sbstr.query_multi(storage_keys)
            querytime = time.time() - tnow

            tnow = time.time()
            clean_balances = {}
            for result in results:
                clean_balances[result[0].params[0]] = {
                    "balance": result[1]["data"]["free"].decode(),
                    "nonce": result[1]["nonce"].decode(),
                    "sent": 0,
                    "received": 0,
                }

            with DATA_LOCK:
                _last_used_block = NEW_BLOCK
                if BALANCES_LIST:
                    for acc, info in clean_balances.items():
                        BALANCES_LIST[acc]["balance"] = info["balance"]
                else:
                    BALANCES_LIST = copy.deepcopy(clean_balances)

            updateglobaltime = time.time() - tnow

        except Exception as e:
            logging.error(f"Error: {e}")
            sbstr = SubstrateInterface(url=rpcurl)
            pass

        logging.info(
            f"BALANCE UPDATE -- StorageKeys:{storagetime:.2f}s - Query:{querytime:.2f}s - GlobalVar:{updateglobaltime:.2f}s"
        )

        while True:
            with DATA_LOCK:
                _new_block = NEW_BLOCK

            if _last_used_block == _new_block:
                time.sleep(0.2)
            else:
                break


def start_balances_list(balance_list: dict, rpcurl, start_wallet=None):
    if start_wallet:
        balance_list[start_wallet] = {
            "balance": None,
            "nonce": None,
        }

    balance_thread = threading.Thread(
        target=update_balances, args=(balance_list, rpcurl), daemon=True
    )
    balance_thread.start()

    while True:
        with DATA_LOCK:
            _balance_list = BALANCES_LIST

        if not _balance_list:
            logging.info(f"Balance not updated yet. Waiting")
            time.sleep(2)
        else:
            break


def start_block_subscription(rpcurl: str):
    block_thread = threading.Thread(
        target=subscribe_blocks, args=(rpcurl,), daemon=True
    )
    block_thread.start()

    while True:
        with DATA_LOCK:
            _new_block = NEW_BLOCK

        if not _new_block:
            logging.info(f"Wait for first new block")
            time.sleep(2)
        else:
            break


def subscribe_blocks(rpcurl: str):
    logging.info("Subscrbe to new blocks")
    sbstr = SubstrateInterface(url=rpcurl)
    sbstr.subscribe_block_headers(new_block)


def new_block(obj, update_nr, subscription_id):
    global NEW_BLOCK
    logging.info(f"New block: #{obj['header']['number']}")
    with DATA_LOCK:
        NEW_BLOCK = int(obj["header"]["number"])


def fast_compose_call(
    call_module: str,
    call_function: str,
    call_params: dict,
    substrate: SubstrateInterface,
):
    call = substrate.runtime_config.create_scale_object(
        type_string="Call", metadata=substrate.metadata
    )

    call.encode(
        {
            "call_module": call_module,
            "call_function": call_function,
            "call_args": call_params,
        }
    )

    return call


def fast_create_signed_extrinsic(
    call,
    signer: Keypair,
    genesis_hash: str,
    substrate: SubstrateInterface,
    signer_nonce: int,
):
    era = "00"
    signature_payload = substrate.runtime_config.create_scale_object(
        "ExtrinsicPayloadValue"
    )
    if "signed_extensions" in substrate.metadata[1][1]["extrinsic"]:
        signature_payload.type_mapping = [["call", "CallBytes"]]
        signed_extensions = substrate.metadata.get_signed_extensions()
        if "CheckMortality" in signed_extensions:
            signature_payload.type_mapping.append(
                ["era", signed_extensions["CheckMortality"]["extrinsic"]]
            )
        if "CheckEra" in signed_extensions:
            signature_payload.type_mapping.append(
                ["era", signed_extensions["CheckEra"]["extrinsic"]]
            )
        if "CheckNonce" in signed_extensions:
            signature_payload.type_mapping.append(
                ["nonce", signed_extensions["CheckNonce"]["extrinsic"]]
            )
        if "ChargeTransactionPayment" in signed_extensions:
            signature_payload.type_mapping.append(
                ["tip", signed_extensions["ChargeTransactionPayment"]["extrinsic"]]
            )
        if "ChargeAssetTxPayment" in signed_extensions:
            signature_payload.type_mapping.append(
                ["asset_id", signed_extensions["ChargeAssetTxPayment"]["extrinsic"]]
            )
        if "CheckMetadataHash" in signed_extensions:
            signature_payload.type_mapping.append(
                ["mode", signed_extensions["CheckMetadataHash"]["extrinsic"]]
            )
        if "CheckSpecVersion" in signed_extensions:
            signature_payload.type_mapping.append(
                [
                    "spec_version",
                    signed_extensions["CheckSpecVersion"]["additional_signed"],
                ]
            )
        if "CheckTxVersion" in signed_extensions:
            signature_payload.type_mapping.append(
                [
                    "transaction_version",
                    signed_extensions["CheckTxVersion"]["additional_signed"],
                ]
            )
        if "CheckGenesis" in signed_extensions:
            signature_payload.type_mapping.append(
                ["genesis_hash", signed_extensions["CheckGenesis"]["additional_signed"]]
            )
        if "CheckMortality" in signed_extensions:
            signature_payload.type_mapping.append(
                ["block_hash", signed_extensions["CheckMortality"]["additional_signed"]]
            )
        if "CheckEra" in signed_extensions:
            signature_payload.type_mapping.append(
                ["block_hash", signed_extensions["CheckEra"]["additional_signed"]]
            )
        if "CheckMetadataHash" in signed_extensions:
            signature_payload.type_mapping.append(
                [
                    "metadata_hash",
                    signed_extensions["CheckMetadataHash"]["additional_signed"],
                ]
            )
    call_data = str(call.data)
    payload_dict = {
        "call": call_data,
        "era": era,
        "nonce": signer_nonce,
        "tip": 0,
        "spec_version": substrate.runtime_version,
        "genesis_hash": genesis_hash,
        "block_hash": genesis_hash,
        "transaction_version": substrate.transaction_version,
        "asset_id": {"tip": 0, "asset_id": None},
        "metadata_hash": None,
        "mode": "Disabled",
    }
    signature_payload.encode(payload_dict)
    if signature_payload.data.length > 256:
        payload_data = ScaleBytes(
            data=blake2b(signature_payload.data.data, digest_size=32).digest()
        )
    else:
        payload_data = signature_payload.data

    signature_version = signer.crypto_type
    signature = signer.sign(payload_data)
    extrinsic = substrate.runtime_config.create_scale_object(
        type_string="Extrinsic", metadata=substrate.metadata
    )
    value = {
        "account_id": f"0x{signer.public_key.hex()}",
        "signature": f"0x{signature.hex()}",
        "call_function": call.value["call_function"],
        "call_module": call.value["call_module"],
        "call_args": call.value["call_args"],
        "nonce": signer_nonce,
        "era": era,
        "tip": 0,
        "asset_id": {"tip": 0, "asset_id": None},
        "mode": "Disabled",
    }
    signature_cls = substrate.runtime_config.get_decoder_class("ExtrinsicSignature")
    if issubclass(signature_cls, substrate.runtime_config.get_decoder_class("Enum")):
        value["signature_version"] = signature_version
    extrinsic.encode(value)

    return extrinsic


def fast_submit_extrinsic(
    extrinsic, substrate: SubstrateInterface, method="author_submitExtrinsic"
):
    request_id = substrate.request_id
    substrate.request_id += 1
    params = [str(extrinsic.data)]

    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": request_id,
    }

    try:
        substrate.websocket.send(json.dumps(payload))
    except WebSocketConnectionClosedException:
        if substrate.config.get("auto_reconnect") and substrate.url:
            # Try to reconnect websocket and retry rpc_request
            substrate.debug_message("Connection Closed; Trying to reconnect...")
            substrate.connect_websocket()

            fast_submit_extrinsic(extrinsic, substrate)
        else:
            # websocket connection is externally created, re-raise exception
            raise


def transfer_all(recipient: str, substrate: SubstrateInterface):
    return fast_compose_call(
        call_module="Balances",
        call_function="transfer_all",
        call_params={"dest": recipient, "keep_alive": False},
        substrate=substrate,
    )


def transfer_to(recipient: str, amount: int, substrate: SubstrateInterface):
    return fast_compose_call(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={"dest": recipient, "value": amount},
        substrate=substrate,
    )


def prepare_distribute(
    start_wallet: Keypair,
    recipients: list[Keypair],
    substrate: SubstrateInterface,
    config: dict,
):
    global BALANCES_LIST
    min_per_recipient = (
        config["spam_wallet_min_distr_amount"] * 10**substrate.token_decimals
    )
    fee = config["fee"] * 10**substrate.token_decimals
    genesis_hash = substrate.get_block_hash(0)
    number_of_recipients = len(recipients)

    with DATA_LOCK:
        start_balance = BALANCES_LIST[start_wallet.ss58_address]["balance"]

    total_fee = int(fee * number_of_recipients)
    wallet_reserve = int(5 * 10**substrate.token_decimals)
    available_balance = start_balance - wallet_reserve - total_fee

    if available_balance <= 0:
        logging.error("No available balance to distribute")
        sys.exit(1)

    logging.info(
        f"Distributing {available_balance / 10**substrate.token_decimals} tokens to {number_of_recipients} recipients"
    )

    amount_to_randomize = available_balance - min_per_recipient * number_of_recipients
    if amount_to_randomize <= 0:
        logging.error(
            f"Not enough token to distribute {min_per_recipient / 10 **substrate.token_decimals} to each account"
        )
        sys.exit(1)

    weights = [random.random() for _ in range(number_of_recipients)]
    total_weight = sum(weights)
    amounts = [
        int(min_per_recipient + (amount_to_randomize * weight / total_weight))
        for weight in weights
    ]

    extrinsics = []
    with DATA_LOCK:
        for recipient in recipients:
            recipient_public_key = "0x" + recipient.public_key.hex()
            call = transfer_to(
                recipient_public_key, amount=amounts.pop(), substrate=substrate
            )
            extrinsics.append(
                fast_create_signed_extrinsic(
                    call=call,
                    signer=start_wallet,
                    genesis_hash=genesis_hash,
                    substrate=substrate,
                    signer_nonce=BALANCES_LIST[start_wallet.ss58_address]["nonce"],
                )
            )
            BALANCES_LIST[start_wallet.ss58_address]["nonce"] += 1

    return extrinsics


def prepare_shuffle(
    accounts: list[Keypair],
    substrate: SubstrateInterface,
    genesis_hash,
    config: dict,
):
    global BALANCES_LIST

    min_balance = config["spam_wallet_min_balance"] * 10**substrate.token_decimals
    fee = config["fee"] * 10**substrate.token_decimals

    _balance_list = {}
    with DATA_LOCK:
        for acc in accounts:
            _balance_list[acc.ss58_address] = BALANCES_LIST[acc.ss58_address].copy()

    _tmp_list = _balance_list.copy()
    acc_c = int(len(accounts) / 2)

    top_sent = sorted(_tmp_list.items(), key=lambda x: x[1]["sent"], reverse=True)[
        :acc_c
    ]

    for key, _ in top_sent:
        _tmp_list.pop(key)

    top_received = sorted(
        _tmp_list.items(), key=lambda x: x[1]["received"], reverse=True
    )[:acc_c]
    # min_send = 10000000
    min_send = 1000
    extrinsics = []
    update_nonce_for = []

    while top_received:
        sender, recipient = top_received.pop(), top_sent.pop()
        se_ba = sender[1]["balance"]

        if se_ba < min_balance + min_send + fee:
            logging.warning(f"Balance to small - Account: {sender[0]}")
            continue

        # Divide by 20 because balance list might be couple blocks behind and potential latency issues
        # max_send = int((se_ba - min_balance - fee) // 20)
        max_send = int(min_send * 2)
        # Because of the //20
        if max_send < min_send:
            max_send = min_send

        amount = random.randint(min_send, max_send)
        sender_kp = next((kp for kp in accounts if kp.ss58_address == sender[0]), None)

        call = transfer_to(recipient[0], amount, substrate)
        extrinsics.append(
            fast_create_signed_extrinsic(
                call=call,
                signer=sender_kp,
                genesis_hash=genesis_hash,
                substrate=substrate,
                signer_nonce=sender[1]["nonce"],
            )
        )
        update_nonce_for.append({"sender": sender, "receiver": recipient})

    with DATA_LOCK:
        for transfer in update_nonce_for:
            BALANCES_LIST[transfer["sender"][0]]["nonce"] += 1
            BALANCES_LIST[transfer["sender"][0]]["sent"] += 1
            BALANCES_LIST[transfer["receiver"][0]]["received"] += 1
            pass
    return extrinsics


def prepare_rmrk(
    accounts: list[Keypair], substrate: SubstrateInterface, rmrk: str, genesis_hash: str
):
    call = fast_compose_call(
        call_module="System",
        call_function="remark",
        call_params={"remark": rmrk},
        substrate=substrate,
    )
    global BALANCES_LIST

    _balance_list = {}
    with DATA_LOCK:
        for acc in accounts:
            _balance_list[acc.ss58_address] = BALANCES_LIST[acc.ss58_address].copy()

    extrinsics = []
    update_nonce_for = []
    skip_item = False
    for acc, item in _balance_list.items():
        if skip_item:
            skip_item = False
            continue

        acc_kp = next((kp for kp in accounts if kp.ss58_address == acc), None)
        extrinsics.append(
            fast_create_signed_extrinsic(
                call=call,
                signer=acc_kp,
                genesis_hash=genesis_hash,
                substrate=substrate,
                signer_nonce=item["nonce"],
            )
        )

        update_nonce_for.append(acc)
        skip_item = True

    with DATA_LOCK:
        for acc in update_nonce_for:
            BALANCES_LIST[acc]["nonce"] += 1

    return extrinsics


def prepare_collect(accounts: list[Keypair], end_wallet, substrate, genesis_hash):
    end_wallet_public_key = "0x" + end_wallet.public_key.hex()

    extrinsics = []
    call = transfer_all(end_wallet_public_key, substrate)
    with DATA_LOCK:
        for account in accounts:
            acc_balance = BALANCES_LIST[account.ss58_address]
            if acc_balance["balance"] > 0:
                # logging.info(f"{account.ss58_address}")
                extrinsics.append(
                    fast_create_signed_extrinsic(
                        call=call,
                        signer=account,
                        genesis_hash=genesis_hash,
                        substrate=substrate,
                        signer_nonce=acc_balance["nonce"],
                    )
                )
                BALANCES_LIST[account.ss58_address]["nonce"] += 1

    return extrinsics


def collect_spam_accounts(config: dict):
    substr = SubstrateInterface(url=config["rpc"][0])
    substr.init_runtime()

    start_wallet = get_start_wallet(substr.ss58_format)
    accounts, balance_list = get_spam_wallets(
        substr.ss58_format, config["spam_wallet_count"], config["derivation"]
    )

    start_block_subscription(config["rpc"][0])
    start_balances_list(balance_list, config["rpc"][0], start_wallet.ss58_address)

    collecting_extrinsics = prepare_collect(
        accounts, start_wallet, substr, substr.get_block_hash(0)
    )

    logging.info("Collecting tokens from all wallets and reaping accounts")
    count = 0
    for extrinsic in collecting_extrinsics:
        fast_submit_extrinsic(extrinsic, substr)
        count += 1
        time.sleep(0.001)

        if count > 900:
            logging.info("More than 900 accounts, sleep 12 seconds")
            time.sleep(12)
            count = 0

    logging.info(f"Submitted {len(collecting_extrinsics)} collecting extrinsics")
    time.sleep(18)  # Wait to make sure everything gets onchain
    count = count_funded_spam_acc(BALANCES_LIST, wallet=start_wallet)
    # if count > 0:
    #     logging.info(
    #         f"{count_funded_spam_acc(BALANCES_LIST, wallet=start_wallet)} spam accounts are STILL funded - sending some token"
    #     )
    #     with DATA_LOCK:
    #         for account in accounts:
    #             acc_balance = BALANCES_LIST[account.ss58_address]
    #             if acc_balance["balance"] > 0:
    #                 call = substr.compose_call(
    #                     call_module="Balances",
    #                     call_function="transfer_keep_alive",
    #                     call_params={
    #                         "dest": account.ss58_address,
    #                         "value": config["spam_wallet_min_balance"]
    #                         * 10**substr.token_decimals,
    #                     },
    #                 )

    #                 extrinsic = substr.create_signed_extrinsic(
    #                     call=call, keypair=start_wallet
    #                 )
    #                 substr.submit_extrinsic(extrinsic, wait_for_inclusion=True)

    #     time.sleep(12)


def distribute_spam_accounts(config: dict):
    substr = SubstrateInterface(url=config["rpc"][0])
    substr.init_runtime()

    start_wallet = get_start_wallet(substr.ss58_format)
    accounts, balance_list = get_spam_wallets(
        substr.ss58_format, config["spam_wallet_count"], config["derivation"]
    )

    start_block_subscription(config["rpc"][0])
    start_balances_list(balance_list, config["rpc"][0], start_wallet.ss58_address)
    distribution_extrinsics = prepare_distribute(start_wallet, accounts, substr, config)

    if distribution_extrinsics:
        logging.info("Distributing tokens from start wallet")
        count = 0
        for extrinsic in distribution_extrinsics:
            fast_submit_extrinsic(extrinsic, substr)
            count += 1
            time.sleep(0.001)

            if count > 300:
                logging.info("Sleep 12 seconds because distributed over 300")
                time.sleep(12)
                count = 0

        logging.info(
            f"Submitted {len(distribution_extrinsics)} distribution extrinsics"
        )
    else:
        logging.error("No Tokens distributed to spam accounts - Check issue")
        sys.exit(1)

    time.sleep(12)  # Wait to make sure everything gets onchain
    logging.info(
        f"{count_funded_spam_acc(BALANCES_LIST, wallet=start_wallet)} spam accounts are funded"
    )


def return_start_wallet_funds(config: dict):
    substr = SubstrateInterface(url=config["rpc"][0])
    start_wallet = get_start_wallet(substr.ss58_format)

    call = substr.compose_call(
        call_module="Balances",
        call_function="transfer_all",
        call_params={"dest": config["cold_wallet"], "keep_alive": False},
    )
    extrinsic = substr.create_signed_extrinsic(call=call, keypair=start_wallet)
    substr.submit_extrinsic(extrinsic)


def send_rmrk(
    rmrk: str, substrate: SubstrateInterface, accounts: list[Keypair], genesis_hash: str
):
    with DATA_LOCK:
        _last_block = NEW_BLOCK
        _new_block = NEW_BLOCK

    extrinsics = prepare_rmrk(accounts, substrate, rmrk, genesis_hash)

    while True:
        if _last_block == _new_block:
            time.sleep(0.1)
        else:
            break

        with DATA_LOCK:
            _new_block = NEW_BLOCK

    for extrinsic in extrinsics:
        fast_submit_extrinsic(extrinsic, substrate)

    logging.info(f"RMRK -- Submitted rmrks")


def spam_thread(
    accounts: list[Keypair],
    config: dict,
    threadcount: int,
    genesis_hash,
    substr: SubstrateInterface,
):
    iteration = 0
    max_iteration = config.get("blocks_to_spam", None)

    logging.info(f"Thread {threadcount} -- Start Shuffle")

    if config.get("rmrk", None):
        send_rmrk(config["rmrk"], substr, accounts, genesis_hash)

    with DATA_LOCK:
        _last_block = NEW_BLOCK
        _new_block = NEW_BLOCK

    while True:
        now = time.time()

        shuffling_extrinsics = prepare_shuffle(accounts, substr, genesis_hash, config)
        shuffle_prep = time.time() - now

        if not shuffling_extrinsics:
            logging.info(f"Thread {threadcount} -- No Funds left to spam")
            break

        while True:
            if _last_block == _new_block:
                time.sleep(0.1)
            else:
                break

            with DATA_LOCK:
                _new_block = NEW_BLOCK

        now = time.time()
        for extrinsic in shuffling_extrinsics:
            fast_submit_extrinsic(extrinsic, substr)

        _last_block = _new_block
        iteration += 1

        tsubmit_extr = time.time() - now
        logging.info(
            f"Thread {threadcount} -- Shuffle: {shuffle_prep:.2f}s - Submit: {tsubmit_extr:.2f}s - Extrinsics: {len(shuffling_extrinsics)}"
        )

        if (shuffle_prep + tsubmit_extr) >= 6:
            logging.warning(
                f"Over 6 seconds to prepare and publish. Not spamming at capacity"
            )

        if max_iteration is not None and iteration >= max_iteration:
            logging.info(f"Thread {threadcount} -- Reached max iterations, exiting")
            break


def start_spammening(config: dict):
    logging.info(f"Start spammening")

    # Needs to be created outside the threads
    substrates = []
    logging.info("Create substrate connections")
    for rpc in config["rpc"]:
        sub = SubstrateInterface(url=rpc)
        sub.init_runtime()
        substrates.append(sub)

    genesis_hash = substrates[0].get_block_hash(0)
    spam_accounts, balance_list = get_spam_wallets(
        substrates[0].ss58_format, config["spam_wallet_count"], config["derivation"]
    )

    start_block_subscription(config["subscription_rpc"])
    start_balances_list(balance_list, config["subscription_rpc"])

    with DATA_LOCK:
        logging.info(f"{count_funded_spam_acc(BALANCES_LIST)} spam accounts are funded")

    split_accounts = split_list(spam_accounts, len(config["rpc"]))

    # spam_thread(split_accounts[0], config, 0, genesis_hash, substrates[0])
    spam_threads = []
    for index, spa in enumerate(split_accounts):
        sp_thread = threading.Thread(
            target=spam_thread,
            args=(spa, config, index, genesis_hash, substrates[index]),
            daemon=True,
        )
        sp_thread.start()
        spam_threads.append(sp_thread)

    for thread in spam_threads:
        thread.join()

    logging.info("Threads done")
