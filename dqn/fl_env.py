import time
from utils.parameter_counter import parameter_count
from sparse_compressor.scheduler import fusion_ratio_scheduler
import numpy as np
from utils.configer import Configer
from client_manager import client_manager
from sparse_optimizer.warmup import warmup_scheduler


class fl_env:
    def __init__(self, config: Configer, record_batch=10):
        self.config = config
        self.record_batch = record_batch
        self.client_manager = None
        self.communication_round = 0
        self.traffic = 0
        self.uncompress_traffic = 0
        self.uncompress_model_size = 0


        self.reward_loss_weight = 0.8

        self.discount_factor = 0.99
        ###########################
        self.last_avg_loss = None
        self.last_avg_acc = None
        self.last_traffic = 0

        self.writer = None

    def reset(self, writer, executor, gpus):
        w_scheduler = warmup_scheduler(start_lr=self.config.trainer.get_start_lr(),
                                       max_lr=self.config.trainer.get_max_lr(),
                                       min_lr=self.config.trainer.get_min_lr(),
                                       base_step=self.config.trainer.get_base_step(),
                                       end_step=self.config.trainer.get_end_step())

        self.client_manager = client_manager(config=self.config,
                                             warmup_scheduler=w_scheduler,
                                             writer=writer,
                                             executor=executor,
                                             available_gpu=gpus)
        
        self.writer = writer
        net = self.client_manager.set_init_mdoel()
        self.uncompress_model_size = parameter_count(net)
        self.traffic = (parameter_count(net) * self.config.general.get_nodes())
        self.last_traffic = self.traffic

        self.communication_round = 0
        self.last_avg_loss = None
        self.last_avg_acc = None
        self.last_traffic = 0
        return 0, 0

    def step(self, action):
        # overwrite fusion_ratio
        action = round(action * 0.05, 3)
        for c in self.client_manager.clients:
            c.fusion_ratio_scheduler = fusion_ratio_scheduler(self.config.trainer.get_max_iteration(), [action])

        tmp_acc = []
        tmp_loss = []
        for i in range(self.record_batch):
            test_acc, test_loss = self.training()
            tmp_acc.append(test_acc)
            tmp_loss.append(test_loss)
            self.communication_round += 1

        avg_loss = sum(tmp_loss) / len(tmp_loss)
        tmp_acc = sum(tmp_acc) / len(tmp_acc)

        next_state = (self.communication_round, avg_loss)

        if self.last_avg_loss is not None:
            # reward = loss 下降％數 + overheads 變化量％數
            reward = ((self.last_avg_loss - avg_loss) / self.last_avg_loss) * self.reward_loss_weight + \
                     (1-((self.traffic - self.last_traffic) / self.uncompress_traffic)) * (1.0 - self.reward_loss_weight)
        else:
            reward = (1-((self.traffic - self.last_traffic) / self.uncompress_traffic)) * (1.0 - self.reward_loss_weight)

        self.last_traffic = self.traffic
        self.last_avg_loss = avg_loss
        self.uncompress_traffic = 0

        done = True if self.communication_round >= self.config.trainer.get_max_iteration() - 1 else False
        if self.writer is not None:
            self.writer.add_scalar("reward", reward, global_step=self.communication_round, walltime=None)
        return next_state, reward, done

    def training(self):
        self.client_manager.set_communication_round(communication_round=self.communication_round)
        # sample dataset
        sampled_client_id = self.client_manager.sample_client()
        self.client_manager.sample_data()
        ####################################################################################################
        trained_gradients = self.client_manager.train()
        # clients transmit to server
        self.traffic += sum([parameter_count(g) for g in trained_gradients])
        self.uncompress_traffic += sum([self.uncompress_model_size for _ in trained_gradients])
        # aggregate
        aggregated_gradient = self.client_manager.aggregate(trained_gradients=trained_gradients)
        # server transmit to clients
        self.traffic += parameter_count(aggregated_gradient) * self.config.general.get_nodes()
        self.uncompress_traffic += self.uncompress_model_size * self.config.general.get_nodes()
        # one step update
        self.client_manager.one_step_update(aggregated_gradient=aggregated_gradient)

        test_acc, test_loss = self.client_manager.global_test()
        print("Test acc: {}, loss: {}".format(test_acc, test_loss))
        if self.writer is not None:
            self.writer.add_scalar("traffic(number_of_parameters)", self.traffic, global_step=self.communication_round, walltime=None)
        return test_acc, test_loss
