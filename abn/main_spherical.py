
import numpy as np
import datasets.utils 
import datasets.wiggles

import torch

import matplotlib.pyplot as plt
import seaborn as sns

import torch.optim as optim
import pandas as pd
from models.spherical_vae import SphericalVAE
from trainer.svae_trainer import SphericalVAETrainer
from datasets.cyclic_walk import CyclicWalk
from datasets.wiggles import Wiggles
from datasets.data_loader import TrainValLoader

import neural_metric
import main_eval

import wandb


# Load dataset

dataset = Wiggles(n_times=1500, n_wiggles=6, amp_wiggles=0.3, embedding_dim=4, noise_var = 0.0001)



"""
Normalize dataset
Min 0 Max 1
"""
dataset.data = dataset.data - dataset.data.min()
dataset.data = dataset.data / dataset.data.max()

data_loader = TrainValLoader(batch_size=20)
data_loader.load(dataset)



model = SphericalVAE(input_dim=4,
                     encoder_dims=[100, 30, 20, 10],
                     latent_dim=2,
                     distribution="vmf")

optimizer = optim.Adam(model.parameters(), lr=0.001)

trainer = SphericalVAETrainer(model=model, optimizer=optimizer)


trainer.train(data_loader, epochs=100)



"""
Visualize Latent Space
"""

with torch.no_grad():
    (z_mean, z_var), (q_z, p_z), z = model.to_latent(dataset.data)

df = pd.DataFrame({"z0": z[:, 0], 
                    "z1": z[:, 1], 
                    "pos": dataset.labels % (2 * np.pi)})

plt.figure(figsize=(10, 10))
sns.scatterplot(x="z0", y="z1", hue="pos", data=df, palette="rainbow")

plt.savefig("results/fig.png")


def get_immersion(model):

    def immersion(theta):
        z = torch.tensor([torch.cos(theta),torch.sin(theta)])
        return model.decode(z)

    return immersion


immersion = get_immersion(model)

metric = neural_metric.NeuralMetric(
            dim=1, embedding_dim=4, immersion=immersion)


base_points = torch.linspace(0,2*torch.pi,1000)

mean_curv_vectors = [metric.mean_curvature(base_point) for base_point in base_points]


mean_curv_norm = [torch.linalg.norm(vec) for vec in mean_curv_vectors]


torch.save(mean_curv_norm, "results/mean_curv_norm.pt")

main_eval.plot(base_points,mean_curv_norm)














