class config_sgc:
    def __init__(self, config):
        self.config = dict(config._sections["sgc"])

    def get_compress_rate(self) -> list:
        cr = self.config["compress_rate"]
        if cr[0] == "[" and cr[-1] == "]":
            cr = cr[1:-1].split(",")
            cr = [float(i) for i in cr]
        else:
            cr = [float(cr)]
        return cr

    def get_local_momentum(self) -> float:
        return float(self.config["local_momentum"])

    def get_approximation_momentum(self) -> float:
        return float(self.config["approximation_momentum"])

    def get_server_momentum(self) -> float:
        return float(self.config["server_momentum"])
