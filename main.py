import sys
import os
from dotenv import load_dotenv
from src.train import TrainServer
import argparse
import requests
import json


load_dotenv()
QUEUE_HOST = os.environ.get("QUEUE_HOST")
MAIN_SERVER_ENDPOINT = os.environ.get("MAIN_SERVER_ENDPOINT")


if __name__ == '__main__':
    try:
        EXCHANGE_TRANSFER_PHOTO = os.environ.get("EXCHANGE_TRANSFER_PHOTO")
        EXCHANGE_UPDATE_MODEL = os.environ.get("EXCHANGE_UPDATE_MODEL")
        train_server = TrainServer(epoch=5, lambda_style=5, lambda_content=5, check_point_after=5, lr=5, snapshot_path=5)
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            sys.exit(0)
