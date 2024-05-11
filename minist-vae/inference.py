# reference: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py
import torch
from vae import VAE

def sample(num_samples,latent_dim):
    """
    Samples from the latent space and return the corresponding
    image space map.
    :param num_samples: (Int) Number of samples
    :return: (Tensor)
    """
    z = torch.randn(num_samples,
                    latent_dim)
    vae = VAE()
    samples = vae.decoder(z)
    return samples

def generate(x):
    """
    Given an input image x, returns the reconstructed image
    :param x: (Tensor) [B x C x H x W]
    :return: (Tensor) [B x C x H x W]
    """
    vae = VAE()
    return vae(x)

if __name__ == "__main__":
    sample()