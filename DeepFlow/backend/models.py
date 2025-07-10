import torch
import torch.nn as nn
import torch.nn.functional as F
import config

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

class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)  

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)   
        return mu + eps * std        

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        recon = self.fc5(h)
        return recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

class VaDE(nn.Module):
    def __init__(self, input_dim, latent_dim, n_clusters):
        super(VaDE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_clusters = n_clusters

        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 128)
        self.fc4 = nn.Linear(128, 256)
        self.fc5 = nn.Linear(256, input_dim)

        self.pi_prior = torch.ones(n_clusters) / n_clusters
        self.mu_c = nn.Parameter(torch.randn(n_clusters, latent_dim))
        self.logvar_c = nn.Parameter(torch.zeros(n_clusters, latent_dim))

    def encode(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc3(z))
        h = F.relu(self.fc4(h))
        return self.fc5(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar, z

    def compute_gamma(self, z):
        z_expand = z.unsqueeze(1)
        mu_c_expand = self.mu_c.unsqueeze(0)
        logvar_c_expand = self.logvar_c.unsqueeze(0)
        var_c_expand = torch.exp(logvar_c_expand)

        log_pi = torch.log(self.pi_prior.to(config.DEVICE) + 1e-10)
        log_prob = -0.5 * torch.sum(
            logvar_c_expand + (z_expand - mu_c_expand) ** 2 / var_c_expand, dim=2
        ) + log_pi
        log_prob = log_prob - torch.logsumexp(log_prob, dim=1, keepdim=True)
        gamma = torch.exp(log_prob)
        return gamma

