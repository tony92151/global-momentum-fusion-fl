import configparser
from utils.compressor.dgc import config_dgc
from utils.compressor.sgc import config_sgc
from utils.compressor.gfdgc import config_gfdgc
from utils.compressor.gfgc import config_gfgc


algo_we_provide = ["dgc", "sgc", "gfdgc", "gfgc"]


class config_general:
    def __init__(self, config):
        self.config = dict(config._sections["general"])

    def get_logdir(self) -> str:
        return self.config["logdir"]

    def get_nodes(self) -> int:
        return int(self.config["nodes"])

    def get_frac(self) -> float:
        return float(self.config["frac"])


class config_trainer:
    def __init__(self, config):
        self.config = dict(config._sections["trainer"])

    def get_device(self):
        return self.config["device"]

    def get_model(self) -> str:
        return self.config["model"]

    def get_dataset(self) -> str:
        return self.config["dataset"]

    def get_dataset_path(self):
        return self.config["dataset_path"]

    def get_local_bs(self) -> int:
        return int(self.config["local_bs"])

    def get_local_ep(self) -> int:
        return int(self.config["local_ep"])

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


class config_compression:
    def __init__(self, config):
        self.config = dict(config._sections["compression"])

    def get_algorithm(self) -> str:
        algo = str(self.config["algorithm"])
        if algo not in algo_we_provide:
            raise ValueError("config: compression -> algorithm, got error config.")
        return algo


class Configer:
    def __init__(self, configfile):
        self.config = configparser.ConfigParser()
        self.config.read(configfile)

        self.general = config_general(self.config)
        self.trainer = config_trainer(self.config)
        self.eval = config_eval(self.config)
        self.agg = config_agg(self.config)

        self.compression = config_compression(self.config)

        if self.compression.get_algorithm() == "dgc":
            try:
                self.dgc = config_dgc(self.config)
            except KeyError:
                print("config read: skip dgc")
        # elif self.compression.get_algorithm() == "sgc":
        #     try:
        #         self.gf = config_sgc(self.config)
        #     except KeyError:
        #         print("config read: skip gf")
        # elif self.compression.get_algorithm() == "gfdgc":
        #     try:
        #         self.gfgc = config_gfdgc(self.config)
        #     except KeyError:
        #         print("config read: skip gfgc")
        # elif self.compression.get_algorithm() == "gfgc":
        #     try:
        #         self.sgc = config_gfgc(self.config)
        #     except KeyError:
        #         print("config read: skip sgc")
