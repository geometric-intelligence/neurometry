import numpy as np
import torch

from neurometry.geometry.curvature.losses import latent_regularization_loss


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def test_latent_regularization_loss():
    config = AttrDict({"dataset_name": "experimental"})
    z = torch.tensor(
        [
            [np.cos(np.pi / 3), np.sin(np.pi / 3)],
            [np.cos(np.pi / 5), np.sin(np.pi / 5)],
            [np.cos(np.pi / 7), np.sin(np.pi / 7)],
        ]
    )
    labels = torch.tensor([180 / 3, 180 / 5, 180 / 7])
    print(z.shape)
    print(labels.shape)
    loss = latent_regularization_loss(labels, z, config).numpy()
    assert np.allclose(loss, 0.0)
