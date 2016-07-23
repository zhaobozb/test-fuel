"""train.py"""
import argparse
import time
from contextlib import contextmanager

from fuel.streams import ServerDataStream

from parallel_server import create_data_stream


@contextmanager
def timer(name):
    """Times a block of code and prints the result.

    Parameters
    ----------
    name : str
        What this block of code represents.

    """
    start_time = time.time()
    yield
    stop_time = time.time()
    print('{} took {} seconds'.format(name, stop_time - start_time))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p', '--parallel', action='store_true',
        help='run data preprocessing in a separate process')
    args = parser.parse_args()

    if args.parallel:
        data_stream = ServerDataStream(('features', ), True)
    else:
        data_stream = create_data_stream(0.005)

    with timer('Training'):
        for i in range(5):
            for data in data_stream.get_epoch_iterator(): time.sleep(0.01)