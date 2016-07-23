"""server.py"""
import time

from fuel.datasets import IndexableDataset
from fuel.schemes import ShuffledScheme
from fuel.server import start_server
from fuel.streams import DataStream
from fuel.transformers import Transformer


class Bottleneck(Transformer):
    """Waits every time data is requested to simulate a bottleneck.

    Parameters
    ----------
    slowdown : float, optional
        Time (in seconds) to wait before returning data. Defaults to 0.

    """
    def __init__(self, *args, **kwargs):
        self.slowdown = kwargs.pop('slowdown', 0)
        super(Bottleneck, self).__init__(*args, **kwargs)

    def get_data(self, request=None):
        if request is not None:
            raise ValueError
        time.sleep(self.slowdown)
        return next(self.child_epoch_iterator)


def create_data_stream(slowdown=0):
    """Creates a bottlenecked data stream of dummy data.

    Parameters
    ----------
    slowdown : float
        Time (in seconds) to wait each time data is requested.

    Returns
    -------
    data_stream : fuel.streams.AbstactDataStream
        Bottlenecked data stream.

    """
    dataset = IndexableDataset({'features': [[0] * 128] * 1000})
    iteration_scheme = ShuffledScheme(examples=1000, batch_size=100)
    data_stream = Bottleneck(data_stream=DataStream.default_stream(dataset=dataset, iteration_scheme=iteration_scheme),
                             slowdown=slowdown)
    return data_stream


if __name__ == "__main__":
    start_server(create_data_stream(0.005))
