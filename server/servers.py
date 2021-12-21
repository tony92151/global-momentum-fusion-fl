from server.base_server import BASE_SERVER

from server.weight_server import weight_server
from server.momentum_server import momentum_server
from utils.configer import Configer


def get_server(con: Configer) -> BASE_SERVER:
    if con.compression.get_algorithm() == "dgc":
        return weight_server
    elif con.compression.get_algorithm() == "sgc":
        return momentum_server
    elif con.compression.get_algorithm() == "gfdgc":
        return weight_server
    elif con.compression.get_algorithm() == "sgfgc":
        return weight_server
