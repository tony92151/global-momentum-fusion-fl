class config_dgc:
    def __init__(self, config):
        self.config = dict(config._sections["dgc"])

    def get_compress_ratio(self) -> float:
        return float(self.config["compress_ratio"])

    def get_momentum(self) -> float:
        return float(self.config["momentum"])

    def get_momentum_correction(self) -> bool:
        if self.config["momentum_correction"] == "False":
            return False
        elif self.config["momentum_correction"] == "True":
            return True
