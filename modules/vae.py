import torch
import torch.nn as nn
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden2mean = nn.Linear(input_dim, latent_dim)
        self.hidden2logv = nn.Linear(input_dim, latent_dim)
        # self.latent2hidden = nn.Linear(latent_dim, input_dim)
        self.reset_params()

    def reset_params(self):
        nn.init.xavier_uniform_(self.hidden2mean.weight)
        nn.init.xavier_uniform_(self.hidden2logv.weight)
        # nn.init.xavier_uniform_(self.latent2hidden.weight)

    def kl_anneal_function(self, anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    def forward(self, h, step, k, x0):
        mean = self.hidden2mean(h)
        logv = self.hidden2logv(h)
        std = torch.exp(0.5 * logv)
        z = torch.randn((h.size(0), h.size(1), self.latent_dim), device=h.device)
        z = z * std + mean
        # restruct_hidden = self.latent2hidden(z)
        # dist_loss = F.mse_loss(restruct_hidden, h)

        # N(mu, sig) ~ N(0, 1) -> -1/2(logσ2-μ2-σ2+1)
        kl_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp()) / logv.size(0)
        kl_weight = self.kl_anneal_function(anneal_function='linear', step=step, k=k, x0=x0)
        # batch size average
        kl_loss = (kl_weight * kl_loss) / h.size(0)
        return kl_loss , z