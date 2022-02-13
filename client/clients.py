from client.base_client import BASE_CLIENT

from client.dgc_client import dgc_client
from client.sgc_client import sgc_client
from client.gfdgc_client import gfdgc_client
from client.gfgc_client import gfgc_client
from client.gmc_client import gmc_client
from client.dgc_gm_client import dgc_gm_client
from utils.configer import Configer


def get_client(con: Configer):
    if con.compression.get_algorithm() == "dgc":
        return dgc_client
    elif con.compression.get_algorithm() == "sgc":
        return sgc_client
    elif con.compression.get_algorithm() == "gfdgc":
        return gfdgc_client
    elif con.compression.get_algorithm() == "gfgc":
        return gfgc_client
    elif con.compression.get_algorithm() == "gmc":
        return gmc_client
    elif con.compression.get_algorithm() == "dgc_gm":
        return dgc_gm_client

