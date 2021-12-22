from client.base_client import BASE_CLIENT


class sgc_client(BASE_CLIENT):
    def __init__(self, cid=None, compressor=None, trainer=None, data=None):
        super(BASE_CLIENT, self).__init__(cid=cid, compressor=compressor, trainer=trainer, data=data)

        self.compress_rate_scheduler = None

    def train(self):
        pass

    def test(self):
        pass
