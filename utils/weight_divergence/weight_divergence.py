# This script is an implement of this paper.
# https://arxiv.org/abs/1806.00582

# In the paper weight_divergence(WD):
#        || W_fedavg - W_SGD ||
# WD =  ------------------------
#             || W_SGD ||

# Rewrite weight_divergence(WD):
#       || (base_model - lr * aggregated_gradient) - (base_model - lr * trainer_gradient) ||
# WD = -------------------------------------------------------------------------------------
#                        || (base_model - lr * trainer_gradient) ||

# Rewrite weight_divergence(WD):
#        lr * || aggregated_gradient - trainer_gradient ||
# WD = --------------------------------------------------------
#            || (base_model - lr * trainer_gradient) ||
import torch
import copy
from utils.aggregator import set_gradient
from utils.opti import SERVEROPTS, FEDOPTS


def weight_divergence(config=None, aggregated_gradient=None, trainer_gradient=None, base_model=None, lr=None,
                      device=torch.device('cpu')):
    if config is None or \
            aggregated_gradient is None or \
            trainer_gradient is None or \
            base_model is None or \
            lr is None:
        raise ValueError("Error value while running weight_divergence function.")

    base_model = copy.deepcopy(base_model)
    aggregated_gradient = copy.deepcopy(aggregated_gradient["gradient"])
    trainer_gradient = copy.deepcopy(trainer_gradient["gradient"])

    aggregated_gradient = [torch.tensor(p) for p in aggregated_gradient]
    trainer_gradient = [torch.tensor(p) for p in trainer_gradient]

    optimizer = SERVEROPTS(config=config, params=base_model.parameters(), lr=lr)
    base_model.cpu().train()
    set_gradient(opt=optimizer, cg=trainer_gradient)
    optimizer.step()

    base_model.parameters()

    W_SGD = []
    for p in base_model.parameters():
        W_SGD.append(p.detach())

    wds = []
    for i, _ in enumerate(aggregated_gradient):
        dv = (lr * torch.norm(torch.add(aggregated_gradient[i].to(device),
                                        trainer_gradient[i].to(device),
                                        alpha=-1.0))) / torch.norm(W_SGD[i].to(device))
        wds.append(dv)

    wds = sum(wds) / len(wds)

    return wds


def weight_divergence_mod(config=None,
                          aggregated_gradient=None,
                          trainer_gradient=None,
                          device=torch.device('cpu')):
    if config is None or \
            aggregated_gradient is None or \
            trainer_gradient is None:
        raise ValueError("Error value while running weight_divergence function.")

    aggregated_gradient = copy.deepcopy(aggregated_gradient["gradient"])
    trainer_gradient = copy.deepcopy(trainer_gradient["gradient"])

    wds = []

    for ag, tg in zip(aggregated_gradient, trainer_gradient):
        wds.append(torch.norm(torch.tensor(ag).to(device).add(torch.tensor(tg).to(device), alpha=-1.0)) /
                   torch.norm(torch.tensor(tg).to(device)))

    return sum(wds) / len(wds)
