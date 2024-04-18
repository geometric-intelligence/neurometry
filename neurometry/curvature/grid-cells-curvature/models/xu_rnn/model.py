""" Representation model of grid cells. """
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
import math
from LSTM import LSTM

@dataclass
class GridCellConfig:
  trans_type: str
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
  adaptive_dr: bool


class GridCell(nn.Module):
  def __init__(self, config: GridCellConfig):
    super().__init__()
    self.encoder = Encoder(config)
    self.decoder = Decoder(config)
    if config.trans_type == 'nonlinear_simple': 
      self.trans = TransNonlinear(config)
    elif config.trans_type == 'lstm': 
      self.trans = TransNonlinear_LSTM(config)

    self.config = config

  def forward(self, data, step):
    config = self.config
    loss_kernel = self._loss_kernel(**data['kernel'])
    
    if config.trans_type == 'nonlinear_simple': 
      loss_trans = self._loss_trans_rnn(**data['trans_rnn'])
    elif config.trans_type == 'lstm': 
      loss_trans = self._loss_trans_lstm(**data['trans_rnn'])

    if self.config.adaptive_dr: 
      loss_isometry = self._loss_isometry_numerical_block(**data['isometry_adaptive'])
    else: 
      loss_isometry = self._loss_isometry_numerical_block(**data['isometry'])
    
    w_reg_u = self.config.w_reg_u - self.config.w_reg_u / config.reg_decay_until * step
    if w_reg_u < 0: 
      w_reg_u = 0

    loss_reg_u = torch.sum(self.decoder.u ** 2) * w_reg_u
    
    loss = loss_kernel + loss_isometry + loss_reg_u + loss_trans

    metrics = {
        'loss_kernel': loss_kernel,
        'loss_trans': loss_trans,
        'loss_isometry': loss_isometry,
        'loss_reg_u': loss_reg_u,
        'loss': loss,
    }

    return loss, metrics

  def path_integration(self, traj):  # traj: [N, T+1, 2]
    dx_traj = torch.diff(traj, dim=1) / self.config.num_grid  # [N, T, 2]
    T = dx_traj.shape[1]

    x_start = traj[:, 0]
    v_start = self.encoder(x_start)
    v_t = {'vanilla': v_start, 'reencode': v_start}
    x_t = {'vanilla': [x_start], 'reencode': [x_start]}
    heatmaps = []
    heatmaps_modules = []

    for t in range(T+1):
      _x_t_vanilla, _heatmaps, _heatmaps_modules = self.decoder.decode(v_t['vanilla'])
      _x_t, _, _ = self.decoder.decode(v_t['reencode'])

      heatmaps.append(_heatmaps)
      x_t['vanilla'].append(_x_t_vanilla)
      x_t['reencode'].append(_x_t)

      if t < T:
        v_t_trans = v_t['vanilla']
        v_t_trans = self.trans(v_t_trans, dx_traj[:, t])
        v_t['vanilla'] = v_t_trans

        # re-encode the last predicted x
        _v_t = self.encoder(_x_t)
        _v_t = self.trans(_v_t, dx_traj[:, t])
        v_t['reencode'] = _v_t

    traj_pred = {k: torch.stack(v[1:], axis=1)
                 for k, v in x_t.items()}  # [N, T, 2]
    heatmaps = torch.stack(heatmaps, axis=1)  # [N, T, H, W]

    err = {'err_' + k: torch.mean(torch.sqrt(torch.sum((v - traj) ** 2, dim=-1))) /
           self.config.num_grid for k, v in traj_pred.items()}

    return {
        'err': err,
        'traj_real': traj,
        'traj_pred': traj_pred,
        'heatmaps': heatmaps,
    }

  def path_integration_lstm(self, traj):  # traj: [N, T+1, 2]
    dx_traj = torch.diff(traj, dim=1) / self.config.num_grid  # [N, T, 2]
    T = dx_traj.shape[1]

    x_start = traj[:, 0]
    v_start = self.encoder(x_start)
    v_t = {'vanilla': v_start, 'reencode': v_start}
    x_t = {'vanilla': [x_start, x_start], 'reencode': [x_start]}
    heatmaps = []
    heatmaps_modules = []

    # without re-encoding
    v_x = self.trans(v_start, dx_traj) # v_x: [N, C], dx: [N, T, 2]
    v_x_trans = v_x # [N, T, C]
    for i in range(T): 
      x_trans, _heatmaps, _heatmaps_modules = self.decoder.decode(v_x_trans[:, i, :])
      x_t['vanilla'].append(x_trans)
      heatmaps.append(_heatmaps)

    for t in range(T+1):
      _x_t, _, _ = self.decoder.decode(v_t['reencode'])
      x_t['reencode'].append(_x_t)

      if t < T:
        # re-encode the last predicted x
        _v_t = self.encoder(_x_t)
        _v_t = self.trans(_v_t, dx_traj[:, t][:, None, :])
        v_t['reencode'] = _v_t

    traj_pred = {k: torch.stack(v[1:], axis=1)
                 for k, v in x_t.items()}  # [N, T, 2]
    heatmaps = torch.stack(heatmaps, axis=1)  # [N, T, H, W]

    err = {'err_' + k: torch.mean(torch.sqrt(torch.sum((v - traj) ** 2, dim=-1))) /
           self.config.num_grid for k, v in traj_pred.items()}

    return {
        'err': err,
        'traj_real': traj,
        'traj_pred': traj_pred,
        'heatmaps': heatmaps,
    }

  def _loss_kernel(self, x, x_prime):
    config = self.config
    if config.w_kernel == 0:
      return torch.zeros([]).to(x.get_device())

    v_x = self.encoder(x)
    u_x_prime = self.decoder(x_prime)

    dx_square = torch.sum(((x - x_prime) / config.num_grid) ** 2, dim=1)
    kernel = torch.exp(- dx_square / (2. * config.sigma ** 2))

    loss_kernel = torch.mean(
        (torch.sum(v_x * u_x_prime, dim=1) - kernel) ** 2) * 30000

    return loss_kernel * config.w_kernel

  def _loss_trans_rnn(self, traj):
    config = self.config
    if config.w_trans == 0:
      return torch.zeros([]).to(x.get_device())

    softmax = torch.nn.Softmax(dim=-1)

    # place cells, x_pc: (0, 1)
    x1 = torch.arange(0,config.num_grid,1).repeat_interleave(config.num_grid)
    x2 = torch.arange(0,config.num_grid,1).repeat(config.num_grid)
    x1 = torch.unsqueeze(x1, 1)
    x2 = torch.unsqueeze(x2, 1)
    x_pc = torch.cat((x1, x2), axis=1) / config.num_grid
    x_pc = x_pc[None, :].cuda(traj.device) #(1, 1600, 2)

    loss_trans = torch.zeros([]).to(traj.get_device())

    v_x = self.encoder(traj[:, 0, :])
    for i in range(traj.shape[1] - 1):
      dist = torch.sum((traj[:, i+1, :][:, None, :] / config.num_grid - x_pc) ** 2, dim=-1)

      y = torch.exp(-dist/(2*self.config.sigma**2)) # (N, 1600)
      dx = (traj[:, i+1, :] - traj[:, i, :]) / config.num_grid

      v_x = self.trans(v_x, dx)
      v_x_trans = v_x

      # get response map
      v_x_trans = v_x_trans[:, None, None, :]
      u = self.decoder.u.permute((1, 2, 0))[None, ...]
      vu = v_x_trans * u  # [N, H, W, C]
      heatmap = vu.sum(dim=-1)
      heatmap_reshape = heatmap.reshape((heatmap.shape[0], -1)) # (N, 1600)
      y_hat = heatmap_reshape

      loss_trans_i = torch.mean(torch.sum((y - y_hat) ** 2, dim=1))
      loss_trans += loss_trans_i

    return loss_trans * config.w_trans
  
  def _loss_trans_lstm(self, traj):
    config = self.config
    if config.w_trans == 0:
      return torch.zeros([]).to(x.get_device())

    softmax = torch.nn.Softmax(dim=-1)

    # place cells, x_pc: (0, 1)
    x1 = torch.arange(0,config.num_grid,1).repeat_interleave(config.num_grid)
    x2 = torch.arange(0,config.num_grid,1).repeat(config.num_grid)
    x1 = torch.unsqueeze(x1, 1)
    x2 = torch.unsqueeze(x2, 1)
    x_pc = torch.cat((x1, x2), axis=1) / config.num_grid
    x_pc = x_pc[None, :].cuda(traj.device) #(1, 1600, 2)

    dist = torch.sum((traj[:, 1:, :].reshape(-1, 2)[:, None, :] / config.num_grid - x_pc) ** 2, dim=-1) # [N*T, 1600]
    dist = dist.reshape(traj.shape[0], traj.shape[1]-1, dist.shape[1]) # [N, T, 1600]
    y = torch.exp(-dist/(2*self.config.sigma**2))

    v_x = self.encoder(traj[:, 0, :])
    dx = (traj[:, 1:, :] - traj[:, :-1, :]) / config.num_grid # [N, T, 2]

    v_x = self.trans(v_x, dx) # v_x: [N, C], dx: [N, T, 2]
    v_x_trans = v_x # [N, T, C]

    # get response map
    v_x_trans = v_x_trans[:, :, None, None, :] # [N, T, 1, 1, C]
    u = self.decoder.u.permute((1, 2, 0))[None, ...] # [1, H, W, C]

    loss_trans = torch.zeros([]).to(traj.get_device())
    for i in range(v_x.shape[1]): 
      v_x_trans_i = v_x_trans[:, i, :] # [N, C]
      v_x_trans_i = v_x_trans_i[:, None, None, :]

      vu = v_x_trans_i * u  # [N, H, W, C]
      heatmap = vu.sum(dim=-1)
      heatmap_reshape = heatmap.reshape((heatmap.shape[0], -1)) # (N, 1600)
      y_hat = heatmap_reshape

      loss_trans_i = torch.mean(torch.sum((y[:, i, :] - y_hat) ** 2, dim=1))
      loss_trans += loss_trans_i
 
    return loss_trans * config.w_trans

  def _loss_isometry_numerical_block(self, x, x_plus_dx1, x_plus_dx2):
    config = self.config
    if config.w_isometry == 0:
      return torch.zeros([]).to(x.get_device())

    num_block = config.num_neurons // config.block_size

    dx_square = torch.sum(((x_plus_dx1 - x) / config.num_grid) ** 2, dim=-1)
    
    if config.adaptive_dr: 
      v_x = self.encoder.get_v_x_adpative(x)
      v_x_plus_dx1 = self.encoder.get_v_x_adpative(x_plus_dx1)
      v_x_plus_dx2 = self.encoder.get_v_x_adpative(x_plus_dx2)
    else: 
      v_x = self.encoder(x).reshape((-1, num_block, config.block_size))
      v_x_plus_dx1 = self.encoder(x_plus_dx1).reshape(
        (-1, num_block, config.block_size))
      v_x_plus_dx2 = self.encoder(x_plus_dx2).reshape(
        (-1, num_block, config.block_size))
      
    loss = torch.zeros([]).to(x.get_device())
    
    for i in range(num_block):
      v_x_i = v_x[:, i]
      v_x_plus_dx1_i = v_x_plus_dx1[:, i]
      v_x_plus_dx2_i = v_x_plus_dx2[:, i]

      inner_pd1 = torch.sum(v_x_i * v_x_plus_dx1_i, dim=-1)
      inner_pd2 = torch.sum(v_x_i * v_x_plus_dx2_i, dim=-1)

      if config.adaptive_dr: 
        loss += torch.sum((num_block * inner_pd1 - num_block * inner_pd2) ** 2 * 0.5 / (0.5 + dx_square[:, i]))
      else: 
        loss += torch.sum((num_block * inner_pd1 - num_block * inner_pd2) ** 2 * 0.5 / (0.5 + dx_square))
    
    return loss * config.w_isometry

  def _loss_isometry_numerical_block_adaptive(self, x, x_plus_dx1, x_plus_dx2):
    # x: (batch_size, num_blocks, 2)
    config = self.config
    if config.w_isometry == 0:
      return torch.zeros([]).to(x.get_device())

    num_block = config.num_neurons // config.block_size
    dx_square = torch.sum(((x_plus_dx1 - x) / config.num_grid) ** 2, dim=-1)

    loss = torch.zeros([]).to(x.get_device())
    for i in range(num_block):
      v_x = self.encoder(x[:, i]).reshape((-1, num_block, config.block_size))
      v_x_plus_dx1 = self.encoder(x_plus_dx1[:, i]).reshape((-1, num_block, config.block_size))
      v_x_plus_dx2 = self.encoder(x_plus_dx2[:, i]).reshape((-1, num_block, config.block_size))

      v_x_i = v_x[:, i] # (batch_size, block_size)
      v_x_plus_dx1_i = v_x_plus_dx1[:, i]
      v_x_plus_dx2_i = v_x_plus_dx2[:, i]

      inner_pd1 = torch.sum(v_x_i * v_x_plus_dx1_i, dim=-1) # (batch_size, )
      inner_pd2 = torch.sum(v_x_i * v_x_plus_dx2_i, dim=-1)

      loss += torch.sum((num_block * inner_pd1 - num_block * inner_pd2) ** 2 * 0.5 / (0.5 + dx_square[:, i]))

    return loss * config.w_isometry


class Encoder(nn.Module):
  def __init__(self, config: GridCellConfig):
    super().__init__()
    self.num_grid = config.num_grid
    self.config = config

    self.v = nn.Parameter(torch.normal(0, 0.001, size=(
        config.num_neurons, self.num_grid, self.num_grid)))  # [C, H, W]

  def forward(self, x):
    v_x = get_grid_code(self.v, x, self.num_grid)
    return v_x

  def get_v_x_adpative(self, x): 
    v_x = get_grid_code_block(self.v, x, self.num_grid, self.config.block_size)
    return v_x


class Decoder(nn.Module):
  def __init__(self, config: GridCellConfig):
    super().__init__()
    self.config = config
    self.u = nn.Parameter(torch.normal(0, 0.001, size=(
        config.num_neurons, config.num_grid, config.num_grid)))

  def forward(self, x_prime):
    u_x_prime = get_grid_code(self.u, x_prime, self.config.num_grid)
    return u_x_prime

  def decode(self, v, quantile=0.995):  # v : [N, C], self.u: [C, H, W]
    config = self.config

    v = v[:, None, None, :]  # [N, 1, 1, C]
    u = self.u.permute((1, 2, 0))[None, ...]  # [1, H, W, C]
    vu = v * u  # [N, H, W, C]
    heatmap = vu.sum(dim=-1)
    heatmap_modules = vu.reshape(
        vu.shape[:3] + (-1, config.block_size)).sum(dim=-1)

    # compute the threshold based on quantile
    heatmap_reshape = heatmap.reshape((heatmap.shape[0], -1))
    _, index = torch.sort(heatmap_reshape, dim=1, descending=True)
    num_selected = int(np.ceil(heatmap_reshape.shape[-1] * (1. - quantile)))
    index = index[:, :num_selected]
    index_x1 = torch.div(index, config.num_grid, rounding_mode='trunc')
    index_x2 = torch.remainder(index, config.num_grid)

    x = torch.stack([torch.median(index_x1, dim=-1).values,
                    torch.median(index_x2, dim=-1).values], dim=-1)

    return x, heatmap, heatmap_modules.permute((0, 3, 1, 2))  # [N, K, H, W]


class TransNonlinear(nn.Module):
  def __init__(self, config: GridCellConfig):
    super().__init__()
    num_blocks = config.num_neurons // config.block_size
    self.config = config

    # initialization
    self.A_modules = nn.Parameter(torch.rand(size=(num_blocks, config.block_size, config.block_size)) * 0.002 - 0.001)
    self.B_modules = nn.Parameter(torch.rand(size=(self.config.num_neurons, 2)) * 0.002 - 0.001)

    self.b = nn.Parameter(torch.zeros(size=[]))

    self.nonlinear = nn.ReLU()

  def forward(self, v, dx):  # v: [N, C], dx: [N, 2]
    num_blocks = self.config.num_neurons // self.config.block_size

    A = torch.block_diag(*self.A_modules)
    B = self.B_modules

    v_x_plus_dx = self.nonlinear(torch.matmul(v, A) + torch.matmul(dx, B.transpose(0, 1)) + self.b)

    return v_x_plus_dx
  
  def _dx_to_theta_id_dr(self, dx):
    theta = torch.atan2(dx[:, 1], dx[:, 0]) % (2 * torch.pi)
    theta_id = torch.floor(theta / (2 * torch.pi / self.config.num_theta)).long()
    dr = torch.sqrt(torch.sum(dx ** 2, axis=-1))

    return theta_id, dr


class TransNonlinear_LSTM(nn.Module):
  def __init__(self, config: GridCellConfig):
    super().__init__()
    self.num_blocks = config.num_neurons // config.block_size
    self.config = config

    self.lstm_list = nn.ModuleList([nn.LSTM(2, config.block_size, 1, batch_first=True) for i in range(self.num_blocks)])
    # self.lstm_list = nn.ModuleList([LSTM(2, config.block_size, 1) for i in range(self.num_blocks)])
  
  def forward(self, v, dx):  # v: [N, C], dx: [N, T, 2]
    v_init = v.reshape(v.shape[0], self.config.block_size, -1)
    v_output = torch.zeros([])

    # dx = dx.permute(1, 0, 2)

    for i in range(self.num_blocks): 
      h_0 = v_init[:, :, i][None, :].to(dx.get_device())
      c_0 = torch.zeros(size=(1, v.shape[0], self.config.block_size)).to(dx.get_device())
      lstm = self.lstm_list[i]
      
      output, (hn, cn) = lstm(dx, (h_0.contiguous(), c_0.contiguous())) # output: [N, T, block_size], hn: [1, N, block_size]

      if i == 0: 
        v_output = output
      else: 
        v_output = torch.cat((v_output, output), 2)
    
    return v_output # v_output: [N, T, C]


def get_grid_code(codebook, x, num_grid):
  # x: [N, 2], range: (-1, 1)
  x_normalized = (x + 0.5) / num_grid * 2. - 1.

  # query the 2D codebook, with bilinear interpolation
  v_x = nn.functional.grid_sample(
      input=codebook.unsqueeze(0).transpose(-1, -2),  # [1, C, H, W]
      grid=x_normalized.unsqueeze(0).unsqueeze(0),  # [1, 1, N, 2]
      align_corners=False,
  )  # [1, C, 1, N]

  v_x = torch.squeeze(torch.squeeze(v_x, 0), 1).transpose(0, 1)  # [N, C]
  # v_x = v_x.squeeze().transpose(0, 1)

  return v_x

def get_grid_code_block(codebook, x, num_grid, block_size):
  # x: [N, num_block, 2], range: (-1, 1)
  x_normalized = (x + 0.5) / num_grid * 2. - 1.

  # query the 2D codebook, with bilinear interpolation
  v_x = nn.functional.grid_sample(
      input=codebook.reshape(-1, block_size, codebook.shape[1], codebook.shape[2]).transpose(-1, -2),  # [num_block, block_size, H, W]
      grid=x_normalized.transpose(0, 1).unsqueeze(1),  # [num_block, 1, N, 2]
      align_corners=False,
  )  # [num_block, block_size, 1, N]
  v_x = v_x.squeeze().permute(2, 0, 1)  # [N, num_block, block_size]

  return v_x

def get_grid_code_int(codebook, x, num_grid):
  # x: [N, 2], range: (-1, 1)
  # codebook: [C, H, W]
  x_normalized = x.long()

  # query the 2D codebook, no interpolation
  v_x = torch.vstack([codebook[:,i,j] for i, j in zip(x_normalized[:,0], x_normalized[:,1])])
  # v_x = v_x.squeeze().transpose(0, 1)  # [N, C]

  return v_x
