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
EXCHANGE_TRAIN_SERVER = os.environ.get("EXCHANGE_TRAIN_SERVER")
ROUTING_KEY_TRAIN = os.environ.get("ROUTING_KEY_TRAIN")

LR = float(os.environ.get("LR"))
LAMBDA_STYLE = float(os.environ.get("LAMBDA_STYLE"))
LAMBDA_CONTENT = float(os.environ.get("LAMBDA_CONTENT"))
EPOCH = int(os.environ.get("EPOCH"))


if __name__ == '__main__':
    try:
        train_server = TrainServer(epoch=EPOCH,
                                   lambda_style=LAMBDA_STYLE,
                                   lambda_content=LAMBDA_CONTENT,
                                   lr=LR,
                                   routing_key=ROUTING_KEY_TRAIN,
                                   queue_host=QUEUE_HOST,
                                   exchange_train_server_name=EXCHANGE_TRAIN_SERVER
                                   )

        train_server.start_work()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            sys.exit(0)
