import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder,self).__init__()
        
        # [batch_size, 28*28] => [batch_szie, 20]
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU(),
        )

        # [batch_size, 20] => [batch_size, 28*28]
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256,28*28),
            nn.Sigmoid()
        )

    def forward(self, x):
        """[summary]

        Args:
            x ([tensor]): shape:[batch_size, 1, 28, 28]
        """
        batch_size = x.shape[0]
        # flatten
        x = x.view(batch_size, -1)
        # encoder
        x = self.encoder(x)
        # decoder
        x = self.decoder(x)
        # reshape
        x = x.view(batch_size, 1, 28, 28)

        return x