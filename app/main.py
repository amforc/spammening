import logging
import sys
import argparse
import yaml
from pathlib import Path
from typing import Dict, Any
from utils import generate_mnemonic, tmp_fix_wallet
from spammening import (
    start_spammening,
    collect_spam_accounts,
    distribute_spam_accounts,
    return_start_wallet_funds,
)


def load_config(network: str) -> Dict[str, Any]:
    logging.info(f"Load config for Network: {network}")
    config_path = Path("config.yaml")

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        all_config = yaml.safe_load(f)

    if network not in all_config:
        raise ValueError(f"Network '{network}' not found in configuration file")

    return all_config[network]


def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    config = load_config(args.network)

    match args.action:
        case "create_addr":
            generate_mnemonic()
        case "spam":
            start_spammening(config)
        case "collect":
            collect_spam_accounts(config)
        case "distribute":
            distribute_spam_accounts(config)
        case "return_funds":
            return_start_wallet_funds(config)
        case "fix":
            tmp_fix_wallet(config)
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spammening")

    parser.add_argument(
        "-n",
        "--network",
        help="The network to use",
        default="paseo",
        choices=["paseo", "kusama", "polkadot", "local", "westend"],
    )
    parser.add_argument(
        "-a",
        "--action",
        help="The action to do",
        default="spam",
        choices=["spam", "create_addr", "collect", "distribute", "return_funds", "fix"],
    )

    args = parser.parse_args()
    main(args)
