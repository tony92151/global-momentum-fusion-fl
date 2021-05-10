class config_gf:
    def __init__(self, config):
        self.config = dict(config._sections["gf"])

    def get_fusing_ratio(self) -> float:
        return float(self.config["fusing_ratio"])

    def get_global_fusion(self) -> bool:
        if self.config["global_fusion"] == "False":
            return False
        elif self.config["global_fusion"] == "True":
            return True

    def get_global_fusion_after_warmup(self) -> bool:
        if self.config["global_fusion_after_warmup"] == "False":
            return False
        elif self.config["global_fusion_after_warmup"] == "True":
            return True


