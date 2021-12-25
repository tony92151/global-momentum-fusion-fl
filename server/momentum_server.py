from server.base_server import BASE_SERVER
import torch
from typing import List
from utils.configer import Configer
from server.aggregater import weight_aggregater


class momentum_server(BASE_SERVER):
    def __init__(self, config: Configer, server_momentun: 0.9, aggregater: None, device: torch.device("cpu")):
        super(momentum_server, self).__init__(config=config, aggregater=aggregater, device=device)

        self.server_momentun = server_momentun
        self.server_m = None

    def update(self, aggregated_gradient):
        if self.server_m is None:
            self.server_m = aggregated_gradient
        else:
            for k in self.aggregated_gradient["gradient"].keys():
                tensor = torch.mul(self.server_m["gradient"][k], self.server_momentun)
                tensor = torch.add(tensor, aggregated_gradient["gradient"][k])
                self.server_m["gradient"][k] = tensor
        return self.server_m

    def aggregate(self, trained_gradients: List[dict]):
        decompressed_gradients = [self.compressor.decompress(gradient) for gradient in trained_gradients]
        aggregated_gradient = self.aggregater(gradient_list=decompressed_gradients, device=self.device)
        m_aggregated_gradient = self.update(aggregated_gradient[0])
        return self.compressor.compress(gradient_dict=m_aggregated_gradient, compress=False)
