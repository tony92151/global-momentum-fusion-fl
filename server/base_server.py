import torch
from typing import List
from utils.configer import Configer
from server.aggregater import weight_aggregater
from sparse_compressor.topk_compressor import topkCompressor


class BASE_SERVER:
    def __init__(self, config: Configer, aggregater: None, device=torch.device("cpu")):
        self.config = config
        self.aggregater = aggregater
        self.device = device
        self.compressor = topkCompressor(compress_rate=1.0)

    def aggregate(self, trained_gradients: List[dict], aggregate_bn=False):
        raise NotImplementedError()
