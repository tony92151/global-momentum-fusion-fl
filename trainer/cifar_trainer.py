import torch
import torch.nn as nn
import time
from utils.configer import Configer
from trainer.base_trainer import BASE_TRAINER
from copy import deepcopy as dcopy

from utils.opti import FEDOPTS, SERVEROPTS
from sparse_compressor.record_SGD import RSGD


class cifar_trainer(BASE_TRAINER):
    def __init__(self, config=None, cid=None, warmup_scheduler=None, device=torch.device("cpu")):
        super(cifar_trainer, self).__init__(config=config, cid=cid, warmup_scheduler=warmup_scheduler, device=device)
        self.verbose = True

    def train_run(self, model=None, data=None, lr=None):
        if model is None:
            model = dcopy(self.model)
        else:
            model = dcopy(model)
            
        model.train().to(self.device)

        optimizer = FEDOPTS(config=self.config, params=model.parameters(), lr=lr)

        eploss = []
        correct = 0
        total_data = 0
        self.print_("client >> cid: {} >> train start, {}".format(self.cid, time.time()))

        for i in range(self.config.trainer.get_local_ep()):
            losses = []
            for data, target in data:
                data = data.to(self.device)
                target = target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                loss = self.loss_function(output, target)
                losses.append(loss.item())
                loss.backward()
                optimizer.step()

                _, preds_tensor = output.max(1)
                correct += preds_tensor.eq(target).sum().item()
                total_data += len(target)

            losses = sum(losses) / len(losses)
            eploss.append(losses)

        self.last_gradient = optimizer.get_last_gradient(model=model)
        ############################################################
        train_loss = sum(eploss) / len(eploss)
        train_acc = correct / total_data
        self.print_("client >> cid: {} >> train done, {}".format(self.cid, time.time()))
        del optimizer
        del model
        return train_acc, train_loss

    def test_run(self, data=None):
        self.print_("client >> cid: {} >> eval start, {}".format(self.cid, time.time()))
        return self.test(data=data)

    def test_global_run(self, data=None):
        return self.test(data=data)

    def test(self, data=None):
        model = dcopy(self.model)
        model.to(self.device)

        losses = []
        correct = 0
        total_data = 0
        for data, target in data:
            data = data.to(self.device)
            target = target.to(self.device)
            output = model(data)
            loss = self.loss_function(output, target)
            losses.append(loss.item())

            _, preds_tensor = output.max(1)
            correct += preds_tensor.eq(target).sum().item()
            total_data += len(target)

        test_loss = sum(losses) / len(losses)
        test_acc = correct / total_data

        return test_acc, test_loss

    def one_step_update(self, aggregated_gradient=None, lr=None):
        if aggregated_gradient["compressed"]:
            raise ValueError("In trainer.one_step_update(), input aggregated_gradient should be un-compressed.")

        self.print_("trainer >> cid: {} >> one_step_update, {}".format(self.cid, time.time()))
        model = dcopy(self.model)
        model.to(self.device).train()
        optimizer = SERVEROPTS(config=self.config, params=model.parameters(), lr=lr)

        self.set_gradient(optimizer=optimizer, uncompressed_aggregate_gradient=aggregated_gradient)
        optimizer.step()
        self.model = dcopy(model)
