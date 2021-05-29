class config_dgc:
    def __init__(self, config):
        self.config = dict(config._sections["dgc"])

    def get_dgc(self) -> bool:
        if self.config["dgc"] == "False":
            return False
        elif self.config["dgc"] == "True":
            return True

    def get_compress_ratio(self) -> list:
        cr = self.config["compress_ratio"]
        if cr[0] == "[" and cr[-1] == "]":
            cr = cr[1:-1].split(",")
            cr = [float(i) for i in cr]
        else:
            cr = [float(cr)]
        return cr

    def get_momentum(self) -> float:
        return float(self.config["momentum"])

    def get_momentum_correction(self) -> bool:
        if self.config["momentum_correction"] == "False":
            return False
        elif self.config["momentum_correction"] == "True":
            return True
