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

class LSTM_shakespeare(torch.nn.Module):
    def __init__(self, n_vocab=80, embedding_dim=3, hidden_dim_1=256, hidden_dim_2=256, nb_layers_1=1, nb_layers_2=1):
        super(LSTM_shakespeare, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.embeddings = torch.nn.Embedding(n_vocab, embedding_dim)
        self.lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim_1, nb_layers_1)
        self.lstm_2 = torch.nn.LSTM(hidden_dim_1, hidden_dim_2, nb_layers_2)
        self.hidden2out = torch.nn.Linear(hidden_dim_2, n_vocab)

    def forward(self, seq_in):
        embeddings = self.embeddings(seq_in.t())
        lstm_out, _ = self.lstm_1(embeddings)
        lstm_out, _ = self.lstm_2(lstm_out)
        ht = lstm_out[-1]
        out = self.hidden2out(ht)
        return out