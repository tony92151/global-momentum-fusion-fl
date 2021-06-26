# This script was modified from
# https://github.com/TalwalkarLab/leaf/blob/master/models/shakespeare/stacked_lstm.py
import torch


# (0.0003, 80, 80, 256), # lr, seq_len, num_classes, num_hidden
class LSTM_shakespeare(torch.nn.Module):
    def __init__(self, n_vocab=80, embedding_dim=3, hidden_dim_1=80, hidden_dim_2=256, nb_layers_1=2,
                 nb_layers_2=1):
        super(LSTM_shakespeare, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim_1 = hidden_dim_1
        self.hidden_dim_2 = hidden_dim_2

        self.embeddings = torch.nn.Embedding(n_vocab, embedding_dim)
        self.lstm_1 = torch.nn.LSTM(embedding_dim, hidden_dim_1, nb_layers_1, dropout=0.5)
        self.lstm_2 = torch.nn.LSTM(hidden_dim_1, hidden_dim_2, nb_layers_2)
        self.hidden2out = torch.nn.Linear(hidden_dim_2, n_vocab)

    def forward(self, seq_in, statas: dict = None):
        if statas is None:
            raise
        embeddings = self.embeddings(seq_in.t())
        lstm_out, state_1 = self.lstm_1(embeddings)
        lstm_out, state_2 = self.lstm_2(lstm_out)
        ht = lstm_out[-1]
        out = self.hidden2out(ht)
        return out, (state_1, state_2)

    def zero_state(self, batch_size):
        return (torch.zeros(1, batch_size, self.embedding_dim), torch.zeros(1, batch_size, self.hidden_dim_1))
