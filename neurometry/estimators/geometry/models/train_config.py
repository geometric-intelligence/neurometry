import torch

# hardware
device = "cuda" if torch.cuda.is_available() else "cpu"

# training
num_epochs = 500
batch_size = 64
lr = 1e-3

recon_weight = 1.0  # weight for the reconstruction loss
kld_weight = 0.03  # 0.03  # weight for KL loss
latent_weight = 10  # weight for latent regularization loss
