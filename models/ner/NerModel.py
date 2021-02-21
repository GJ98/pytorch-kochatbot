import torch
from torch import nn

class NerModel(nn.Module):
    def __init__(self, config):
        super(NerModel, self).__init__()

        self.config = config

        self.embed = nn.Embedding(num_embeddings=self.config['vocab_size'] + 1,
                                  embedding_dim=self.config['embed_size'],
                                  padding_idx=0)
        
        self.lstm = nn.LSTM(input_size=self.config['embed_size'], 
                            hidden_size=self.config['hidden_size'], 
                            num_layers=self.config['lstm_num_layers'],
                            bidirectional=True)

        self.linear = nn.Linear(in_features=2*self.config['hidden_size'],
                                out_features=self.config['tag_size'] + 1)

    def forward(self, x):

        embed = self.embed(x.long())

        lstm, _ = self.lstm(embed)

        lstm_dropout = nn.functional.dropout(lstm, p=0.5)

        linear = self.linear(lstm_dropout)

        return linear

