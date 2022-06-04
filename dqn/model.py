import torch

class DQN(torch.nn.Module):
    def __init__(self, state_size=5, action_size=21):
        super(DQN, self).__init__()
        self.Linear1 = torch.nn.Linear(state_size, state_size*2)
        self.Relu1 = torch.nn.ReLU()
        self.Linear2 = torch.nn.Linear(state_size*2, action_size)
        self.Dropout1 = torch.nn.Dropout()
        self.Linear3 = torch.nn.Linear(action_size, action_size)

    def forward(self, vlaue):
        out = self.Linear1(vlaue)
        out = self.Relu1(out)
        out = self.Linear2(out)
        out = self.Dropout1(out)
        out = self.Linear3(out)
        return out
