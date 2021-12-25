from server.base_server import BASE_SERVER
import torch
from typing import List
from utils.configer import Configer
from server.aggregater import weight_aggregater


class weight_server(BASE_SERVER):
    def __init__(self, config: Configer, aggregater: None, device: torch.device("cpu")):
        super(weight_server, self).__init__(config=config, aggregater=aggregater, device=device)

    def aggregate(self, trained_gradients: List[dict]):
        decompressed_gradients = [self.compressor.decompress(gradient) for gradient in trained_gradients]
        aggregated_gradient = self.aggregater(gradient_list=decompressed_gradients, device=self.device)
        return self.compressor.compress(gradient_dict=aggregated_gradient[0], compress=False)
