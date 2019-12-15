import torch
from booster.utils import prod, batch_reduce
from torch import nn
from torch.distributions import Normal


class SimpleVAE(nn.Module):
    def __init__(self, tensor_shape, z_dim, nhid, nlayers, dropout=0, out_features=None):
        super().__init__()

        # input and output shapes
        spatial_dims = tensor_shape[2:]
        out_dim = tensor_shape[1] if out_features is None else out_features
        self.input_shape = tensor_shape
        self.output_shape = (-1, out_dim, *spatial_dims)

        # latent space
        self.z_dim = z_dim
        self.register_buffer('prior', torch.zeros((1, 2 * z_dim)))

        # inference model
        layers = []
        h = prod(self.input_shape[1:])
        for k in range(nlayers):
            layers += [nn.Linear(h, nhid), nn.Dropout(dropout), nn.ReLU()]
            h = nhid
        self.encoder = nn.Sequential(*layers, nn.Linear(h, 2 * self.z_dim))

        # generative model
        layers = []
        h = z_dim
        for k in range(nlayers):
            layers += [nn.Linear(h, nhid), nn.Dropout(dropout), nn.ReLU()]
            h = nhid
        self.decoder = nn.Sequential(*layers, nn.Linear(h, prod(self.output_shape[1:])))

    def _params(self, logits):
        mu_z, logvar_z = logits.chunk(2, dim=1)
        std_z = logvar_z.mul(0.5).exp()
        return mu_z, std_z

    def forward(self, x, **kwargs):

        # posterior
        qz = self.encoder(x.view(x.size(0), -1))

        mu_qz, std_qz = self._params(qz)
        posterior = Normal(mu_qz, std_qz)

        # prior
        pz = self.prior.expand(x.size(0), -1)
        mu_pz, std_pz = self._params(pz)
        prior = Normal(mu_pz, std_pz)

        # sample z \sim q(z|x)
        z = posterior.rsample()

        # generate logits of p_theta(x | z)
        x_ = self._px(z)

        # compute KL(q(z|x) || p(z))
        kl = batch_reduce(posterior.log_prob(z) - prior.log_prob(z))

        return {'x_': x_, 'z': z, 'qz': posterior, 'pz': prior, 'kl': [kl]}

    def _px(self, z):
        """compute and return p_\theta(x|z)"""
        x_ = self.decoder(z)
        return x_.view(self.output_shape)

    def sample_from_prior(self, N, **kwargs):
        # prior
        pz = self.prior.expand(N, -1)
        mu_pz, std_pz = self._params(pz)
        prior = Normal(mu_pz, std_pz)

        # sample z \sim q(z|x)
        z = prior.rsample()

        x_logits = self._px(z)

        return {'x_': x_logits, 'z': z, 'pz': prior}
