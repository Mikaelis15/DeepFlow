import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

class IDEC(nn.Module):
    def __init__(self, autoencoder, n_clusters, latent_dim=None):
        super(IDEC, self).__init__()
        self.encoder = autoencoder.encoder
        self.decoder = autoencoder.decoder
        self.latent_dim = latent_dim or autoencoder.latent_dim
        self.n_clusters = n_clusters

        self.cluster_layer = nn.Parameter(torch.Tensor(n_clusters, self.latent_dim))
        nn.init.xavier_normal_(self.cluster_layer.data)

        self.regressor = nn.Sequential(
            nn.Linear(self.latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2), 
            nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)

        q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.cluster_layer)**2, dim=2))
        q = q.pow((1 + 1) / 2)
        q = (q.t() / torch.sum(q, dim=1)).t()

        y_pred = self.regressor(z)

        return x_recon, y_pred, q, z