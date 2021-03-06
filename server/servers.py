import torch
from server.base_server import BASE_SERVER

from server.weight_server import weight_server
from server.momentum_server import momentum_server
from utils.configer import Configer
from server.aggregater import weight_aggregater, add_aggregater


def get_server(con: Configer, device=torch.device("cpu")):
    if con.compression.get_algorithm() == "dgc":
        return weight_server(config=con,
                             aggregater=weight_aggregater,
                             device=device)
    elif con.compression.get_algorithm() == "sgc":
        return momentum_server(config=con,
                               server_momentun=con.sgc.get_server_momentum(),
                               aggregater=weight_aggregater,
                               device=device)
    elif con.compression.get_algorithm() == "gfdgc":
        return weight_server(config=con,
                             aggregater=weight_aggregater,
                             device=device)
    elif con.compression.get_algorithm() == "gfgc":
        return weight_server(config=con,
                             aggregater=weight_aggregater,
                             device=device)
    elif con.compression.get_algorithm() == "gmc":
        return weight_server(config=con,
                             aggregater=add_aggregater,
                             device=device)
    elif con.compression.get_algorithm() == "dgc_gm":
        return momentum_server(config=con,
                               server_momentun=con.dgc_gm.get_server_momentum(),
                               aggregater=weight_aggregater,
                               device=device)
