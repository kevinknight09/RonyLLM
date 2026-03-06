import torch
import torch.nn as nn

class MiniGPT(nn.Module):

    def __init__(self, vocab_size, embed_dim=256, heads=4, layers=4):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers
        )

        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):

        x = self.embedding(x)

        x = self.transformer(x)

        x = self.fc(x)

        return x