"""Representation model of grid cells."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class GridCellConfig:
    freeze_decoder: bool
    trans_type: str  # not needed
    num_grid: int
    num_neurons: int
    block_size: int
    rnn_step: int
    reg_decay_until: int
    sigma: float
    w_kernel: float
    w_trans: float
    w_isometry: float
    w_reg_u: float
    adaptive_dr: bool  # not needed
    s_0: float
    x_saliency: list
    sigma_saliency: float
    reward_step: int
    saliency_type: str


class GridCell(nn.Module):
    def __init__(self, config: GridCellConfig):
        super().__init__()
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.trans = TransformNonlinear(config)

        self.config = config

    def forward(self, data, step):
        config = self.config
        loss_kernel = self._loss_kernel(**data["kernel"])

        loss_transform = self._loss_transform_rnn(**data["trans_rnn"], step=step)

        loss_isometry = self._loss_isometry_numerical_block(**data["isometry_adaptive"])

        w_reg_u = (
            self.config.w_reg_u - self.config.w_reg_u / config.reg_decay_until * step
        )
        if w_reg_u < 0:
            w_reg_u = 0

        loss_reg_u = torch.sum(self.decoder.u**2) * w_reg_u

        loss = loss_kernel + loss_isometry + loss_reg_u + loss_transform

        metrics = {
            "loss_kernel": loss_kernel,
            "loss_trans": loss_transform,
            "loss_isometry": loss_isometry,
            "loss_reg_u": loss_reg_u,
            "loss": loss,
        }

        return loss, metrics

    def path_integration(self, traj):  # traj: [num_traj, T+1, 2]
        """Path integration with the vanilla model.

        Parameters
        ----------
        traj : torch.Tensor, shape [num_traj, T+1, 2]
            num_traj trajectories, each with T+1 time steps.

        Returns
        -------
        dict
            Dictionary containing the following keys:
            - err: dict
                Dictionary containing the errors of the model.
            - traj_real: torch.Tensor, shape [num_traj, T+1, 2]
                Real trajectory.
            - traj_pred: dict
                Dictionary containing the predicted trajectories.
            - activity: dict
                Dictionary containing the activity of the model.

            - heatmaps: torch.Tensor, shape [num_traj, T, resolution, resolution]
        """

        #         list of dictionaries. Each dictionary has the structure:
        # - 'vanilla': float, mean error of vanilla model for path integration step
        # - 'reencode': float, mean error of reencode model for path integration step
        dx_traj = torch.diff(traj, dim=1) / self.config.num_grid  # [num_traj, T, 2]
        T = dx_traj.shape[1]

        x_0 = traj[:, 0]
        r_0 = self.encoder(x_0)
        r_t = {"vanilla": [r_0], "reencode": [r_0]}
        x_t = {"vanilla": [x_0], "reencode": [x_0]}
        heatmaps = []
        # heatmaps_modules = []

        for t in range(T + 1):
            _x_t_vanilla, _heatmaps, _heatmaps_modules = self.decoder.decode(
                r_t["vanilla"][t]
            )
            _x_t_reencode, _, _ = self.decoder.decode(r_t["reencode"][t])

            heatmaps.append(_heatmaps)
            x_t["vanilla"].append(_x_t_vanilla)
            x_t["reencode"].append(_x_t_reencode)

            if t < T:
                r_t_transformed = self.trans(r_t["vanilla"][t], dx_traj[:, t])
                r_t["vanilla"].append(r_t_transformed)

                # re-encode the last predicted x
                _r_t_reencode = self.encoder(_x_t_reencode)
                _r_t_reencode_transformed = self.trans(_r_t_reencode, dx_traj[:, t])
                r_t["reencode"].append(_r_t_reencode_transformed)

        traj_pred = {
            key: torch.stack(value[1:], axis=1) for key, value in x_t.items()
        }  # [num_traj, T, 2]
        activity = {
            key: torch.stack(value, axis=1) for key, value in r_t.items()
        }  # [num_traj, T, num_neurons]
        heatmaps = torch.stack(
            heatmaps, axis=1
        )  # [num_traj, T, resolution, resolution]

        err = {
            "err_"
            + key: torch.sqrt(torch.sum((value - traj) ** 2, dim=-1))
            / self.config.num_grid
            for key, value in traj_pred.items()
        }

        return {
            "err": err,
            "traj_real": traj,
            "traj_pred": traj_pred,
            "activity": activity,
            "heatmaps": heatmaps,
        }

    def _loss_kernel(self, x, x_prime):
        config = self.config
        if config.w_kernel == 0:
            return torch.zeros([]).to(x.get_device())

        v_x = self.encoder(x)
        u_x_prime = self.decoder(x_prime)

        dx_square = torch.sum(((x - x_prime) / config.num_grid) ** 2, dim=1)
        kernel = torch.exp(-dx_square / (2.0 * config.sigma**2))

        loss_kernel = (
            torch.mean((torch.sum(v_x * u_x_prime, dim=1) - kernel) ** 2) * 30000
        )

        return loss_kernel * config.w_kernel

    def _loss_transform_rnn(self, traj, step):
        config = self.config
        if config.w_trans == 0:
            return torch.zeros([]).to(traj.get_device())

        # place cells, x_pc: (0, 1)
        x1 = torch.arange(0, config.num_grid, 1).repeat_interleave(config.num_grid)
        x2 = torch.arange(0, config.num_grid, 1).repeat(config.num_grid)
        x1 = torch.unsqueeze(x1, 1)
        x2 = torch.unsqueeze(x2, 1)
        x_grid = torch.cat((x1, x2), axis=1) / config.num_grid
        x_pc = x_grid[None, :].cuda(traj.device)  # (1, 1600, 2)

        loss_transform = torch.zeros([]).to(traj.get_device())

        v_x = self.encoder(traj[:, 0, :])
        for i in range(traj.shape[1] - 1):
            dist = torch.sum(
                (traj[:, i + 1, :][:, None, :] / config.num_grid - x_pc) ** 2, dim=-1
            )

            y = torch.exp(
                -dist / (2 * self.config.sigma**2)
            )  # (num_traj 1600) -> place field at location given by traj[:, i + 1, :]
            dx = (traj[:, i + 1, :] - traj[:, i, :]) / config.num_grid

            v_x = self.trans(v_x, dx)
            v_x_trans = v_x

            # get response map
            v_x_trans = v_x_trans[:, None, None, :]
            u = self.decoder.u.permute((1, 2, 0))[None, ...]
            vu = v_x_trans * u  # [num_traj, resolution, resolution, num_neurons]
            heatmap = vu.sum(dim=-1)  # [num_traj, resolution, resolution]
            heatmap_reshape = heatmap.reshape(
                (heatmap.shape[0], -1)
            )  # (num_traj, 1600)
            y_hat = heatmap_reshape  # actual "place cell" activity over the grid (linear readout of grid cells)

            saliency_kernel = (
                self._saliency_kernel(x_grid, config.saliency_type)
                .unsqueeze(0)
                .to(traj.device)
            )  # (1, 1600)
            if step < config.reward_step:
                L_error = (y - y_hat) ** 2
            else:
                L_error = saliency_kernel * (y - y_hat) ** 2

            loss_transform_i = torch.mean(torch.sum(L_error, dim=1))
            loss_transform += loss_transform_i

        return loss_transform * config.w_trans

    def _saliency_kernel(self, x_grid, saliency_type):
        if saliency_type == "gaussian":
            return self._saliency_kernel_gaussian(x_grid)
        if saliency_type == "left_half":
            return self._saliency_kernel_left_half(x_grid)
        raise NotImplementedError

    def _saliency_kernel_gaussian(self, x_grid):
        config = self.config
        s_0 = config.s_0
        x_saliency = torch.tensor([config.x_saliency[0], config.x_saliency[1]]).to(
            x_grid.device
        )
        sigma_saliency = config.sigma_saliency

        # Calculate the squared differences, scaled by respective sigma values
        diff = x_grid - x_saliency
        scaled_diff_sq = (diff[:, 0] ** 2 / sigma_saliency**2) + (
            diff[:, 1] ** 2 / sigma_saliency**2
        )

        # Compute the Gaussian function
        normalization_factor = 2 * np.pi * sigma_saliency * sigma_saliency
        s_x = s_0 * torch.exp(-0.5 * scaled_diff_sq) / normalization_factor

        return 1 + s_x

    def _saliency_kernel_left_half(self, x_grid):
        config = self.config
        s_0 = config.s_0
        s_x = s_0 * (x_grid[:, 0] < 0.5).float()
        return 1 + s_x

    def _loss_isometry_numerical_block(self, x, x_plus_dx1, x_plus_dx2):
        config = self.config
        if config.w_isometry == 0:
            return torch.zeros([]).to(x.get_device())

        num_block = config.num_neurons // config.block_size

        dx_square = torch.sum(((x_plus_dx1 - x) / config.num_grid) ** 2, dim=-1)

        v_x = self.encoder.get_v_x_adaptive(x)
        v_x_plus_dx1 = self.encoder.get_v_x_adaptive(x_plus_dx1)
        v_x_plus_dx2 = self.encoder.get_v_x_adaptive(x_plus_dx2)

        loss = torch.zeros([]).to(x.get_device())

        for i in range(num_block):
            v_x_i = v_x[:, i]
            v_x_plus_dx1_i = v_x_plus_dx1[:, i]
            v_x_plus_dx2_i = v_x_plus_dx2[:, i]

            inner_pd1 = torch.sum(v_x_i * v_x_plus_dx1_i, dim=-1)
            inner_pd2 = torch.sum(v_x_i * v_x_plus_dx2_i, dim=-1)

            loss += torch.sum(
                (num_block * inner_pd1 - num_block * inner_pd2) ** 2
                * 0.5
                / (0.5 + dx_square[:, i])
            )

        return loss * config.w_isometry


class Encoder(nn.Module):
    """
    Neural network encoder. This class encodes a 2D position in the grid cell population.

    Attributes
    ----------
    num_grid : int
        Number of positions along each dimension in the discretized 2D environment.
    config : GridCellConfig
        Configuration of the grid cell model.
    v : nn.Parameter, shape [num_neurons, num_grid, num_grid]
        Weights of the encoder.

    Methods
    -------
    forward(x)
        Forward pass of the encoder.
    get_v_x_adaptive(x)
    """

    def __init__(self, config: GridCellConfig):
        super().__init__()
        self.num_grid = config.num_grid
        self.config = config

        self.v = nn.Parameter(
            torch.normal(
                0, 0.001, size=(config.num_neurons, self.num_grid, self.num_grid)
            )
        )  # [num_neurons, resolution, resolution]

    def forward(self, x):
        """Forward pass of the encoder.

        Parameters
        ----------
        x : torch.Tensor, shape [num_traj, 2]
            2D position across N trajectories.

        Returns
        -------
        torch.Tensor, shape [num_traj, num_neurons]
            Activity of the grid cell population across N trajectories.
        """
        return get_grid_code(self.v, x, self.num_grid)

    def get_v_x_adaptive(self, x):

        return get_grid_code_block(self.v, x, self.num_grid, self.config.block_size)


class Decoder(nn.Module):
    def __init__(self, config: GridCellConfig):
        super().__init__()
        self.config = config
        self.u = nn.Parameter(
            torch.normal(
                0, 0.001, size=(config.num_neurons, config.num_grid, config.num_grid)
            )
        )

    def forward(self, x_prime):
        return get_grid_code(self.u, x_prime, self.config.num_grid)

    def decode(
        self, r, quantile=0.995
    ):  # r : [num_traj, num_neurons], self.u: [num_neurons, resolution, resolution]
        config = self.config

        r = r[:, None, None, :]  # [num_traj, 1, 1, num_neurons]
        u = self.u.permute((1, 2, 0))[
            None, ...
        ]  # [1, resolution, resolution, num_neurons]
        ru = r * u  # [num_traj, resolution, resolution, num_neurons]
        heatmap = ru.sum(dim=-1)
        heatmap_modules = ru.reshape(ru.shape[:3] + (-1, config.block_size)).sum(dim=-1)

        # compute the threshold based on quantile
        heatmap_reshape = heatmap.reshape((heatmap.shape[0], -1))
        _, index = torch.sort(heatmap_reshape, dim=1, descending=True)
        num_selected = int(np.ceil(heatmap_reshape.shape[-1] * (1.0 - quantile)))
        index = index[:, :num_selected]
        index_x1 = torch.div(index, config.num_grid, rounding_mode="trunc")
        index_x2 = torch.remainder(index, config.num_grid)

        x = torch.stack(
            [
                torch.median(index_x1, dim=-1).values,
                torch.median(index_x2, dim=-1).values,
            ],
            dim=-1,
        )

        return (
            x,
            heatmap,
            heatmap_modules.permute((0, 3, 1, 2)),
        )  # [num_traj, K, resolution, resolution]


class TransformNonlinear(nn.Module):
    def __init__(self, config: GridCellConfig):
        super().__init__()
        num_blocks = config.num_neurons // config.block_size
        self.config = config

        # initialization
        self.A_modules = nn.Parameter(
            torch.rand(size=(num_blocks, config.block_size, config.block_size)) * 0.002
            - 0.001
        )
        self.B_modules = nn.Parameter(
            torch.rand(size=(self.config.num_neurons, 2)) * 0.002 - 0.001
        )

        self.b = nn.Parameter(torch.zeros(size=[]))

        self.nonlinear = nn.ReLU()

    def forward(self, v, dx):  # v: [num_traj, num_neurons], dx: [num_traj, 2]

        A = torch.block_diag(*self.A_modules)
        B = self.B_modules

        return self.nonlinear(
            torch.matmul(v, A) + torch.matmul(dx, B.transpose(0, 1)) + self.b
        )

    def _dx_to_theta_id_dr(self, dx):
        theta = torch.atan2(dx[:, 1], dx[:, 0]) % (2 * torch.pi)
        theta_id = torch.floor(theta / (2 * torch.pi / self.config.num_theta)).long()
        dr = torch.sqrt(torch.sum(dx**2, axis=-1))

        return theta_id, dr


def get_grid_code(codebook, x, num_grid):
    """
    Get grid code from the codebook.

    Parameters
    ----------
    """
    # x: [num_traj, 2], range: (-1, 1)
    x_normalized = (x + 0.5) / num_grid * 2.0 - 1.0

    # query the 2D codebook, with bilinear interpolation
    v_x = nn.functional.grid_sample(
        input=codebook.unsqueeze(0).transpose(
            -1, -2
        ),  # [1, num_neurons, resolution, resolution]
        grid=x_normalized.unsqueeze(0).unsqueeze(0),  # [1, 1, num_traj, 2]
        align_corners=False,
    )  # [1, num_neurons, 1, N]

    # v_x = v_x.squeeze().transpose(0, 1)

    return torch.squeeze(torch.squeeze(v_x, 0), 1).transpose(
        0, 1
    )  # [num_traj, num_neurons]


def get_grid_code_block(codebook, x, num_grid, block_size):
    # x: [N, num_block, 2], range: (-1, 1)
    x_normalized = (x + 0.5) / num_grid * 2.0 - 1.0

    # query the 2D codebook, with bilinear interpolation
    v_x = nn.functional.grid_sample(
        input=codebook.reshape(
            -1, block_size, codebook.shape[1], codebook.shape[2]
        ).transpose(
            -1, -2
        ),  # [num_block, block_size, resolution, resolution]
        grid=x_normalized.transpose(0, 1).unsqueeze(1),  # [num_block, 1, num_traj, 2]
        align_corners=False,
    )  # [num_block, block_size, 1, N]
    return v_x.squeeze().permute(2, 0, 1)  # [num_traj, num_block, block_size]
