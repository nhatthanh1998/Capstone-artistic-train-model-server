import sys
from src.train import TrainServer


if __name__ == '__main__':
    try:
        train_server = TrainServer()
        train_server.start_work()
    except KeyboardInterrupt:
        print('Interrupted')
        try:
            sys.exit(0)
        except SystemExit:
            sys.exit(0)