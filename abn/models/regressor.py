"""Regressor."""

import torch
import torch.nn.functional as F


class Regressor(torch.nn.Module):
    """Regressor.
    
    Parameters
    ----------
    input_dim : int 
        Dimension of input data.
        Example: Dimension of the latent space of VAE.
    h_dim : int
        Width of hidden layers.
    output_dim : int
        Dimension of output data. 
        Example: If we are estimating an angle phi, 
        we would need to estimate cos(phi), sin(phi) 
        => output_dim = 2
    """

    def __init__(self, input_dim=2, h_dim=20, output_dim=4):
        super(Regressor, self).__init__()
        self.input_layer = torch.nn.Linear(input_dim, h_dim)
        self.layer1 = torch.nn.Linear(h_dim, h_dim)
        self.layer2 = torch.nn.Linear(h_dim, h_dim)
        self.output_layer = torch.nn.Linear(h_dim, output_dim)

    def forward(self, x):
        """Predict.

        Parameters
        ----------
        x : array-like, shape=[batch_size, input_dim]
            Input data (independent variable).
            Example: latent variables of VAE.
        
        Returns
        -------
        x : array-like shape= [batch_size, output_dim]
            Output (dependent variable).
            Example: estimated head direction angle.
        """
        x = F.relu(self.input_layer(x))
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.tanh(self.output_layer(x))
        return x
