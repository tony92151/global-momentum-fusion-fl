# This script was modified from
# https://github.com/TalwalkarLab/leaf/blob/master/models/shakespeare/stacked_lstm.py
import torch


# (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
# class LSTM_shakespeare(torch.nn.Module):
#     def __init__(self, n_vocab=80, embedding_dim=80, hidden_dim_1=256, hidden_dim_2=256,
#                  nb_layers_1=4, nb_layers_2=1, dropout=0.2):
#         super(LSTM_shakespeare, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.hidden_dim_1 = hidden_dim_1
#         self.hidden_dim_2 = hidden_dim_2

#         self.embeddings = torch.nn.Embedding(n_vocab, embedding_dim)
#         self.lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim_1, nb_layers_1, dropout=dropout)
#         self.lstm_2 = torch.nn.LSTM(hidden_dim_1, hidden_dim_2, nb_layers_2)
#         self.hidden2out = torch.nn.Linear(hidden_dim_2, n_vocab)

#     def forward(self, seq_in):
#         embeddings = self.embeddings(seq_in.t())
#         lstm_out, _ = self.lstm_1(embeddings)
#         lstm_out, _ = self.lstm_2(lstm_out)
#         ht = lstm_out[-1]
#         out = self.hidden2out(ht)
#         return out

class LSTM_shakespeare_2L(torch.nn.Module):
    def __init__(self, n_vocab=80, embedding_dim=64, hidden_dim_1=256, hidden_dim_2=256, nb_layers_1=1, nb_layers_2=1):
        super(LSTM_shakespeare_2L, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.embeddings = torch.nn.Embedding(n_vocab, embedding_dim)
        self.lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim_1, nb_layers_1)
        self.lstm_2 = torch.nn.LSTM(hidden_dim_1, hidden_dim_2, nb_layers_2)
        self.hidden2out = torch.nn.Linear(hidden_dim_2, n_vocab)

    def forward(self, seq_in, state=None):
        if state is not None:
            embeddings = self.embeddings(seq_in.t())
            lstm_out, h_state1 = self.lstm_1(embeddings, state["h1"])
            lstm_out, h_state2 = self.lstm_2(lstm_out, state["h2"])
        else:
            embeddings = self.embeddings(seq_in.t())
            lstm_out, h_state1 = self.lstm_1(embeddings)
            lstm_out, h_state2 = self.lstm_2(lstm_out)
        ht = lstm_out[-1]
        out = self.hidden2out(ht)
        return out, {"h1": (h_state1[0].clone().detach(), h_state1[1].clone().detach()),
                     "h2": (h_state2[0].clone().detach(), h_state2[1].clone().detach())}

    def zero_state(self, batch_size, device=torch.device('cpu')):
        zero_state = {
            "h1": (torch.zeros(1, batch_size, self.hidden_dim_1).to(device),
                   torch.zeros(1, batch_size, self.hidden_dim_1).to(device)),
            "h2": (torch.zeros(1, batch_size, self.hidden_dim_2).to(device),
                   torch.zeros(1, batch_size, self.hidden_dim_2).to(device))
        }
        return zero_state


class LSTM_shakespeare_1L(torch.nn.Module):
    def __init__(self, n_vocab=80, embedding_dim=64, hidden_dim_1=256, nb_layers_1=1):
        super(LSTM_shakespeare_1L, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1

        self.embeddings = torch.nn.Embedding(n_vocab, embedding_dim)
        self.lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim_1, nb_layers_1)
        self.hidden2out = torch.nn.Linear(hidden_dim_1, n_vocab)

    def forward(self, seq_in, state=None):
        if state is not None:
            embeddings = self.embeddings(seq_in.t())
            lstm_out, h_state1 = self.lstm_1(embeddings, state["h1"])
        else:
            embeddings = self.embeddings(seq_in.t())
            lstm_out, h_state1 = self.lstm_1(embeddings)
        ht = lstm_out[-1]
        out = self.hidden2out(ht)

        return out, {"h1": (h_state1[0].clone().detach(), h_state1[1].clone().detach())}

    def zero_state(self, batch_size, device=torch.device('cpu')):
        zero_state = {
            "h1": (torch.zeros(1, batch_size, self.hidden_dim_1).to(device),
                   torch.zeros(1, batch_size, self.hidden_dim_1).to(device))
        }
        return zero_state
