from trainer.cifar_trainer import cifar_trainer
from trainer.sha_trainer import sha_trainer
from utils.configer import Configer


def get_trainer(con: Configer):
    if con.trainer.get_dataset() == "cifar10":
        return cifar_trainer
    elif con.trainer.get_dataset() == "shakespeare":
        return sha_trainer
