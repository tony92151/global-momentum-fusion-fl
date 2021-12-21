class config_gf:
    def __init__(self, config):
        self.config = dict(config._sections["gf"])

    def get_fusing_ratio(self) -> list:
        fr = self.config["fusing_ratio"]
        if fr[0] == "[" and fr[-1] == "]":
            fr = fr[1:-1].split(",")
            fr = [float(i) for i in fr]
        else:
            fr = [float(fr)]
        return fr

    def get_compress_ratio(self) -> list:
        fr = self.config["fusing_ratio"]
        if fr[0] == "[" and fr[-1] == "]":
            fr = fr[1:-1].split(",")
            fr = [float(i) for i in fr]
        else:
            fr = [float(fr)]
        return fr

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

    def get_fusion_momentum(self) -> float:
        return float(self.config["fusion_momentum"])

