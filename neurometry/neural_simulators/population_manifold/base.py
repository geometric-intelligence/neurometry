import torch
import os
os.environ["GEOMSTATS_BACKEND"] = "pytorch"
import geomstats.backend as gs


class PopulationManifold(torch.nn.Module):
    """
    A base class for populations of neurons with receptive fields on a manifold.

    This class provides common functionality for simulating populations of neurons
    whose preferred directions are distributed on a manifold, using the von Mises-Fisher distribution.
    """

    def __init__(self, n_neurons, manifold, nonlinearity="relu", ref_frequency=200, fano_factor=1.0):
        """
        Initialize the PopulationManifold.

        Args:
            n_neurons (int): Number of neurons in the population.
            manifold: The manifold object (e.g., Hypersphere or Torus from geomstats).
            nonlinearity (str): Must be one of 'relu', 'sigmoid', 'tanh', or 'linear'.
            ref_frequency (float, optional): Reference frequency of noisy points. Default is 200 Hz.
            fano_factor (float, optional): Fano factor defined as the variance divided by the mean. Default is 1.
        """
        super().__init__()
        
        self.n_neurons = n_neurons
        self.manifold = manifold
        self.nonlinearity = nonlinearity
        self.ref_frequency = ref_frequency
        self.fano_factor = fano_factor
        self._gen_projection_matrix()

    def _gen_projection_matrix(self):
        """Generate a random encoding matrix.

        Returns
        -------
        projection_matrix : array-like, shape=[self.manifold.embedding_space.dim, self.n_neurons]
            Random encoding matrix.
        """
        self.projection_matrix = gs.random.uniform(-1, 1, (self.manifold.embedding_space.dim, self.n_neurons))

    def _encode_points(self, states):
        """Encode points on a manifold using a given encoding matrix.

        Parameters
        ----------
        states : array-like, shape=[num_points, manifold_extrinsic_dim]
            Points on the manifold.

        Returns
        -------
        encoded_points : array-like, shape=[num_points, encoding_dim]
            Encoded points.
        """
        return states @ self.projection_matrix
    
    def _apply_nonlinearity(self, encoded_points, nonlinearity):
        """Apply a nonlinearity to the encoded points.

        Inputs
        ----------
        encoded_points : array-like, shape=[num_points, encoding_dim]
            Encoded points.
        nonlinearity : str
            Nonlinearity to apply. Must be one of 'relu', 'sigmoid', 'tanh', or 'linear'.

        Returns
        -------
        nonlinearity_points : array-like, shape=[num_points, encoding_dim]
            Encoded points after applying the nonlinearity.
        """
        if nonlinearity == "relu":
            return torch.nn.functional.relu(encoded_points)
        if nonlinearity == "sigmoid":
            return torch.nn.functional.sigmoid(encoded_points)
        if nonlinearity == "tanh":
            return torch.nn.functional.tanh(encoded_points)
        if nonlinearity == "linear":
            return encoded_points
        raise ValueError("Nonlinearity Must be one of 'relu', 'sigmoid', 'tanh', or 'linear'")

    def _gaussian_spikes(self, firing_rates, fano_factor):
        """Generate Gaussian spike trains from data.
        This a good approximation for the Poisson spike trains when the mean firing rate is high.

        Inputs
        ----------
        firing_rates : array-like, shape=[num_points, num_neurons]
            Points on the underlying manifold.
        fano_factor : float, optional
            Fano factor defined as the variance divided by the mean. Default is 1.

        Returns
        -------
        spikes : array-like, shape=[num_points, num_neurons]
            Gaussian spike trains.
        """
        std = torch.sqrt(firing_rates * fano_factor)
        return torch.normal(firing_rates, std)
        
    def forward(self, states):
        encoded_points = self._encode_points(states)
        neural_response = self.ref_frequency * self._apply_nonlinearity(encoded_points, self.nonlinearity)
        noisy_response = self._gaussian_spikes(neural_response, self.fano_factor)
        return noisy_response
        


