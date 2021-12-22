from client.base_client import BASE_CLIENT


class gfgc_client(BASE_CLIENT):
    def __init__(self, cid=None, compressor=None, trainer=None, data=None):
        super(BASE_CLIENT, self).__init__(cid=cid, compressor=compressor, trainer=trainer, data=data)
        self.compress_rate_scheduler = None
        self.fusion_ratio_scheduler = None

    def train(self):
        self.trainer.train_run(data=self.sampled_data)
        self.last_gradient = None
        pass

    def test(self):
        pass

    def one_step_update(self, aggregated_gradient=None):
        raise NotImplementedError()