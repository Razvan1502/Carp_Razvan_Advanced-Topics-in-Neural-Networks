import json
import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    train(config)
