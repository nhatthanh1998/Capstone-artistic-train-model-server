import os
import sys

from dotenv import load_dotenv
from src.train import TrainServer


load_dotenv()
QUEUE_HOST = os.environ.get("QUEUE_HOST")
MAIN_SERVER_ENDPOINT = os.environ.get("MAIN_SERVER_ENDPOINT")


if __name__ == '__main__':
    try:
        train_server = TrainServer(queue_host=QUEUE_HOST, main_server_endpoint=MAIN_SERVER_ENDPOINT)
        train_server.start_work()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            sys.exit(0)