class config_dgc_gm:
    def __init__(self, config):
        self.config = dict(config._sections["dgc_gm"])

    def get_compress_rate(self) -> list:
        cr = self.config["compress_rate"]
        if cr[0] == "[" and cr[-1] == "]":
            cr = cr[1:-1].split(",")
            cr = [float(i) for i in cr]
        else:
            cr = [float(cr)]
        return cr

    def get_momentum(self) -> float:
        return float(self.config["momentum"])

    def get_server_momentum(self) -> float:
        return float(self.config["server_momentum"])

