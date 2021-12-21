class config_gfdgc:
    def __init__(self, config):
        self.config = dict(config._sections["gfdgc"])

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

    def get_fusing_ratio(self) -> list:
        fr = self.config["fusing_ratio"]
        if fr[0] == "[" and fr[-1] == "]":
            fr = fr[1:-1].split(",")
            fr = [float(i) for i in fr]
        else:
            fr = [float(fr)]
        return fr

    def get_global_momentum(self) -> float:
        return float(self.config["global_momentum"])
