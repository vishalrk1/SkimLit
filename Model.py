import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def gather_last_relevant_hidden(hiddens, seq_lens):
    """Extract and collect the last relevant
    hidden state based on the sequence length."""
    seq_lens = seq_lens.long().detach().cpu().numpy() - 1
    out = []
    for batch_index, column_index in enumerate(seq_lens):
        out.append(hiddens[batch_index, column_index])
    return torch.stack(out)


class SkimlitModel(nn.Module):
    def __init__(self, embedding_dim, vocab_size, hidden_dim, n_layers, linear_output, num_classes, pretrained_embeddings=None, padding_idx=0):
        super(SkimlitModel, self).__init__()

        # Initalizing embeddings
        if pretrained_embeddings is None:
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=pretrained_embeddings, padding_idx=padding_idx)

        # LSTM layers
        self.lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, bidirectional=True)

        # FC layers
        self.fc_text = nn.Linear(2*hidden_dim, linear_output)

        self.fc_line_num = nn.Linear(20, 64)
        self.fc_total_line = nn.Linear(24, 64)

        self.fc_final = nn.Linear((64+64+linear_output), num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        x_in, seq_lens, line_nums, total_lines = inputs
        x_in = self.embeddings(x_in)

        # RNN outputs 
        out, b_n = self.lstm1(x_in)
        x_1 = gather_last_relevant_hidden(hiddens=out, seq_lens=seq_lens)

        # FC layers output
        x_1 = F.relu(self.fc_text(x_1))
        x_2 = F.relu(self.fc_line_num(line_nums))
        x_3 = F.relu(self.fc_total_line(total_lines))

        x = torch.cat((x_1, x_2, x_3), dim=1)
        x = self.dropout(x)
        x = self.fc_final(x)
        return x