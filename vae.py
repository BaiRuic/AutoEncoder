import torch.nn as nn
import numpy as np
import torch

class Vari_AutoEncoder(nn.Module):
    def __init__(self):
        super(Vari_AutoEncoder,self).__init__()
        
        # [batch_size, 28*28] => [batch_szie, 20]
        # u [batc_size, 10]
        # sigma [batch_size, 10]
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
            nn.Linear(10, 64),
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
        # [batch_size, 20] 包括均值和期望  各十个
        h_ = self.encoder(x) 

        # [batch_size, 20] => [batch_size, 10] and [batch_size, 10]
        mu, sigma = h_.chunk(2,dim=1) 
        # reparamtrize trich
        h = mu + sigma * torch.randn_like(sigma)
        # kld 公式、
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) + 
            torch.pow(sigma, 2) - 
            torch.log(1e-8 + torch.pow(sigma,2)) - 1 
        ) / np.prod(x.shape)

        # decoder
        x = self.decoder(h)
        # reshape
        x = x.view(batch_size, 1, 28, 28)

        return x, kld