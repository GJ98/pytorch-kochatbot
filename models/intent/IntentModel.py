import torch
from torch import nn



class IntentModel(nn.Module):
    def __init__(self, config):
        super(IntentModel, self).__init__()
        self.embed = nn.Embedding(num_embeddings=config['vocab_size'] + 1, 
                                  embedding_dim=config['embed_size'],
                                  padding_idx=0)

        self.embed_dropout = nn.Dropout(p=config['dropout_prob'])

        self.conv1 = nn.Conv1d(in_channels=config['embed_size'], out_channels=128, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=config['embed_size'], out_channels=128, kernel_size=4)
        self.conv3 = nn.Conv1d(in_channels=config['embed_size'], out_channels=128, kernel_size=5)
        
        self.pool1 = nn.MaxPool1d(kernel_size=config['maxlen'] - 2)
        self.pool2 = nn.MaxPool1d(kernel_size=config['maxlen'] - 3)
        self.pool3 = nn.MaxPool1d(kernel_size=config['maxlen'] - 4)

        self.hidden1 = nn.Linear(384, 128)
        self.hidden_dropout = nn.Dropout(p=config['dropout_prob'])
        self.hidden2 = nn.Linear(128, config['intent_size'])

    def forward(self, x):
        x_embed = self.embed(x.long())

        x_embed_dropout = self.embed_dropout(x_embed)
        x_embed_dropout = torch.einsum('ijk->ikj', x_embed_dropout)
        
        x_conv1 = nn.functional.relu(self.conv1(x_embed_dropout))
        x_conv2 = nn.functional.relu(self.conv2(x_embed_dropout))
        x_conv3 = nn.functional.relu(self.conv3(x_embed_dropout))

        pool1 = self.pool1(x_conv1)
        pool2 = self.pool2(x_conv2)
        pool3 = self.pool3(x_conv3)

        pool1 = torch.einsum('ijk->ij', pool1)
        pool2 = torch.einsum('ijk->ij', pool2)
        pool3 = torch.einsum('ijk->ij', pool3)

        concat = torch.cat((pool1, pool2, pool3), 1)

        hidden1 = nn.functional.relu(self.hidden1(concat))

        hidden_dropout = self.hidden_dropout(hidden1)

        hidden2 = self.hidden2(hidden_dropout)

        return hidden2






