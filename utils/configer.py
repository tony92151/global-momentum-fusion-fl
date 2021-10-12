import configparser
from utils.compressor.dgc import config_dgc
from utils.compressor.gf import config_gf
from utils.compressor.gfgc import config_gfgc
from utils.compressor.sgc import config_sgc


class config_general:
    def __init__(self, config):
        self.config = dict(config._sections["general"])

    def get_tbpath(self) -> str:
        return self.config["tbpath"]

    def get_nodes(self) -> int:
        return int(self.config["nodes"])


class config_trainer:
    def __init__(self, config):
        self.config = dict(config._sections["trainer"])

    def get_device(self):
        return self.config["device"]
    
    def get_model(self) -> str:
        return self.config["model"]

    def get_dataset_type(self):
        return self.config["dataset_type"]

    def get_dataset_path(self):
        return self.config["dataset_path"]

    def get_local_bs(self) -> int:
        return int(self.config["local_bs"])

    def get_local_ep(self) -> int:
        return int(self.config["local_ep"])

    def get_frac(self) -> float:
        return float(self.config["frac"])

    def get_lr(self) -> float:
        return float(self.config["lr"])

    def get_optimizer(self):
        return self.config["optimizer"]

    def get_optimizer_args(self) -> dict:
        try:
            args = eval(self.config["optimizer_args"])
        except KeyError:
            args = {}
        return args

    def get_lossfun(self):
        return self.config["lossfun"]

    def get_max_iteration(self) -> int:
        return int(self.config["max_iteration"])

    # warmup
    def get_start_lr(self) -> float:
        return float(self.config["start_lr"])

    def get_max_lr(self) -> float:
        return float(self.config["max_lr"])

    def get_min_lr(self) -> float:
        return float(self.config["min_lr"])

    def get_base_step(self) -> int:
        return int(self.config["base_step"])

    def get_end_step(self) -> int:
        return int(self.config["end_step"])


class config_eval:
    def __init__(self, config):
        self.config = dict(config._sections["eval"])

    def get_output(self):
        return self.config["output"]

    def get_title(self):
        return self.config["title"]


class config_agg:
    def __init__(self, config):
        self.config = dict(config._sections["aggregator"])

    def get_threshold(self) -> int:
        return int(self.config["threshold"])

    def get_optimizer(self) -> str:
        opts = ["SGD", "ADAGRAD", "ADAM", "YOGI"]
        if self.config["optimizer"] not in opts:
            raise ValueError("Optimizer in aggregator should in {}".format(opts))
        return self.config["optimizer"]

    def get_optimizer_args(self) -> dict:
        try:
            args = eval(self.config["optimizer_args"])
        except KeyError:
            args = {}
        return args


class Configer:
    def __init__(self, configfile):
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

        self.general = config_general(self.config)
        self.trainer = config_trainer(self.config)
        self.eval = config_eval(self.config)
        self.agg = config_agg(self.config)

        try:
            self.dgc = config_dgc(self.config)
        except KeyError:
            print("config read: skip dgc")

        try:
            self.gf = config_gf(self.config)
        except KeyError:
            print("config read: skip gf")

        try:
            self.gfgc = config_gfgc(self.config)
        except KeyError:
            print("config read: skip gfgc")

        try:
            self.sgc = config_sgc(self.config)
        except KeyError:
            print("config read: skip sgc")

