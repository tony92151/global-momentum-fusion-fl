class config_sgc:
    def __init__(self, config):
        self.config = dict(config._sections["sgc"])

    def get_local_momentum(self) -> float:
        return float(self.config["local_momentum"])

    def get_approximation_momentum(self) -> float:
        return float(self.config["approximation_momentum"])

    def get_global_momentum(self) -> float:
        return float(self.config["global_momentum"])
