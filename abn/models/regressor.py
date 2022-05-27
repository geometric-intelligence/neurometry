"""Regressor."""

import torch
import torch.nn.functional as F


class Regressor(torch.nn.Module):
    """Regressor."""

    def __init__(self, input_dim=2, h_dim=20, output_dim=2):
        super(Regressor, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, h_dim)
        self.layer1 = torch.nn.Linear(h_dim, h_dim)
        self.layer2 = torch.nn.Linear(h_dim, h_dim)
        self.output_layer = torch.nn.Linear(h_dim, output_dim)

    def forward(self, x):
        """Predict."""
        x = F.relu(self.input_layer(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.output_layer(x))
        return x
