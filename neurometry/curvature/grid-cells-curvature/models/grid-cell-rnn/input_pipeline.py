""" Data Generator """
import ml_collections
import numpy as np


class TrainDataset:
  def __init__(self, config: ml_collections.ConfigDict, model_config: ml_collections.ConfigDict):
    self.config = config
    self.num_grid = model_config.num_grid
    self.trans_type = model_config.trans_type
    self.rnn_step = model_config.rnn_step
    self.num_blocks = model_config.num_neurons // model_config.block_size
    self.adaptive_dr = model_config.adaptive_dr

    self.dx_list = self._generate_dx_list(config.max_dr_trans)
    # self.dx_list = self._generate_dx_list_continous(config.max_dr_trans)
    self.scale_vector = np.zeros(self.num_blocks) + config.max_dr_isometry

  def __iter__(self):
    while True:
      trans_data_rnn = self._gen_data_trans_rnn()

      if self.adaptive_dr: 
        iso_data_adaptive = self._gen_data_iso_numerical_adaptive()
        yield {
          "kernel": self._gen_data_kernel(),
          "trans_rnn": trans_data_rnn,
          "isometry_adaptive": iso_data_adaptive,
        }
      else: 
        iso_data = self._gen_data_iso_numerical()
        yield {
            "kernel": self._gen_data_kernel(),
            "trans_rnn": trans_data_rnn,
            "isometry": iso_data,
        }

  def _gen_data_kernel(self):
    batch_size = self.config.batch_size
    config = self.config

    theta = np.random.random(size=int(batch_size * 1.5)) * 2 * np.pi
    dr = np.abs(np.random.normal(size=int(batch_size * 1.5))
                * config.sigma_data) * self.num_grid
    dx = _dr_theta_to_dx(dr, theta)

    x_max = np.fmin(self.num_grid - 0.5, self.num_grid - 0.5 - dx)
    x_min = np.fmax(-0.5, -0.5 - dx)
    select_idx = np.where((x_max[:, 0] > x_min[:, 0]) & (
        x_max[:, 1] > x_min[:, 1]))[0][:batch_size]
    x_max, x_min, dx = x_max[select_idx], x_min[select_idx], dx[select_idx]
    assert len(dx) == batch_size

    x = np.random.random(size=(batch_size, 2)) * (x_max - x_min) + x_min
    x_prime = x + dx

    return {'x': x, 'x_prime': x_prime}

  def _gen_data_trans_rnn(self):
    # if it is not for visualization, we try our best to sample the trajectories
    # uniformly wihtin the whole region.
    n_traj = 100
    n_steps = self.rnn_step
    dx_list = self.dx_list

    dx_idx = np.random.choice(len(dx_list), size=[n_traj * 10, n_steps])
    dx = dx_list[dx_idx]  # [N, T, 2]
    dx_cumsum = np.cumsum(dx, axis=1)  # [N, T, 2]

    # sample starting positions of trajectories
    x_start_max = np.fmin(
        self.num_grid - 3, np.min(self.num_grid - 2 - dx_cumsum, axis=1))
    x_start_min = np.fmax(3, np.max(-dx_cumsum + 2, axis=1))  # [N, 2]

    # we choose those traj where x_start_max > x_start_min
    select_idx = np.where(np.sum(x_start_max <= x_start_min, axis=1) == 0)[0]
    assert len(select_idx) >= n_traj
    select_idx = select_idx[:n_traj]
    x_start_max, x_start_min = x_start_max[select_idx], x_start_min[select_idx]
    dx_cumsum = dx_cumsum[select_idx]
    x_start = np.random.random((n_traj, 2)) * \
        (x_start_max - x_start_min) + x_start_min
    x_start = x_start[:, None]  # [N, 1, 2]
    x_start = np.round(x_start - 0.5)

    traj = np.concatenate((x_start, x_start + dx_cumsum), axis=1) # [N, T+1, 2]

    return {'traj': traj}

  def _gen_data_iso_numerical(self):
    batch_size = self.config.batch_size
    config = self.config

    theta = np.random.random(size=(batch_size, 2)) * 2 * np.pi
    dr = np.sqrt(np.random.random(size=(batch_size, 1))) * \
        config.max_dr_isometry
    dx = _dr_theta_to_dx(dr, theta)  # [N, 2, 2]

    x_max = np.fmin(self.num_grid - 0.5,
                    np.min(self.num_grid - 0.5 - dx, axis=1))
    x_min = np.fmax(-0.5, np.max(-0.5 - dx, axis=1))
    x = np.random.random(size=(batch_size, 2)) * (x_max - x_min) + x_min
    x_plus_dx1 = x + dx[:, 0]
    x_plus_dx2 = x + dx[:, 1]

    return {'x': x, 'x_plus_dx1': x_plus_dx1, 'x_plus_dx2': x_plus_dx2}

  def _gen_data_iso_numerical_adaptive(self):
    batch_size = self.config.batch_size # // 5
    num_blocks = self.num_blocks
    config = self.config

    theta = np.random.random(size=(batch_size, num_blocks, 2)) * 2 * np.pi # (batch_size, num_blocks, 2)
    dr = np.sqrt(np.random.random(size=(batch_size, num_blocks, 1))) * np.tile(self.scale_vector, (batch_size, 1))[:, :, None] # (batch_size, num_blocks, 1)
    dx = _dr_theta_to_dx(dr, theta)  # [N, num_blocks, 2, 2]

    x_max = np.fmin(self.num_grid - 0.5,
                    np.min(self.num_grid - 0.5 - dx, axis=2))
    x_min = np.fmax(-0.5, np.max(-0.5 - dx, axis=2))
    x = np.random.random(size=(batch_size, num_blocks, 2)) * (x_max - x_min) + x_min # (batch_size, num_blocks, 2)
    x_plus_dx1 = x + dx[:, :, 0]
    x_plus_dx2 = x + dx[:, :, 1]

    return {'x': x, 'x_plus_dx1': x_plus_dx1, 'x_plus_dx2': x_plus_dx2}

  def _generate_dx_list(self, max_dr, interval=1.):
    dx_list = []
    max_dx_int = int(np.ceil(max_dr) + 1)
    for i in np.arange(0, max_dx_int, interval):
      for j in np.arange(0, max_dx_int, interval):
        if np.sqrt(i ** 2 + j ** 2) <= max_dr:
          dx_list.append(np.array([i, j]))
          if i > 0:
            dx_list.append(np.array([-i, j]))
          if j > 0:
            dx_list.append(np.array([i, -j]))
          if i > 0 and j > 0:
            dx_list.append(np.array([-i, -j]))
    dx_list = np.stack(dx_list)
    dx_list = dx_list.astype(np.float32)

    return dx_list

  def _generate_dx_list_continous(self, max_dr):
    dx_list = []
    batch_size = self.config.batch_size

    dr = np.sqrt(np.random.random(size=(batch_size,))) * max_dr
    np.random.shuffle(dr)
    theta = np.random.random(size=(batch_size,)) * 2 * np.pi
    
    dx = _dr_theta_to_dx(dr, theta)

    return dx


def _dr_theta_to_dx(dr, theta):
  dx_x = dr * np.cos(theta)
  dx_y = dr * np.sin(theta)
  dx = np.stack([dx_x, dx_y], axis=-1)

  return dx


class EvalDataset:
  def __init__(self, config: ml_collections.ConfigDict, max_dr, num_grid):
    self.config = config
    self.num_grid = num_grid
    # for evaluation, we sample trajectories where the positions
    # are all integers.
    self.dx_list = self._generate_dx_list(max_dr)

  def __iter__(self):
    while True:
      yield {
          "traj_vis": self._gen_trajectory_vis(self.config.n_traj_vis, self.config.n_inte_step_vis),
          "traj": self._gen_trajectory(self.config.n_traj, self.config.n_inte_step),
      }

  def _gen_trajectory_vis(self, n_traj, n_steps):
    # For visualization, we don't want the trajectory to be overlapped too much.
    # Thus, we generate trajectories starting from the top-left (i.e., [5, 5]),
    # and slowly moving to the bottom-right.
    dx_list = self.dx_list

    x_start = np.reshape([5, 5], newshape=(1, 1, 2))  # [1, 1, 2]
    dx_idx_pool = np.where((dx_list[:, 0] >= -1) & (dx_list[:, 1] >= -1))[0]
    # dx_idx_pool = np.where((dx_list[:, 0] >= 0) & (dx_list[:, 1] >= -1))[0]
    dx_idx = np.random.choice(
        dx_idx_pool, size=[n_traj * 50, n_steps])
    dx = dx_list[dx_idx]
    dx_cumsum = np.cumsum(dx, axis=1)  # [N, T, 2]

    # traj: [N, T+1, 2]
    traj = np.concatenate(
        (np.tile(x_start, [dx_cumsum.shape[0], 1, 1]), x_start + dx_cumsum), axis=1)

    # the trajectories shoudn't go beyond the border
    select_idx = np.where(np.sum(traj >= self.num_grid, axis=(1, 2)) == 0)[0]
    assert len(select_idx) >= n_traj
    select_idx = select_idx[:n_traj]
    traj = traj[select_idx]

    return {'traj': traj}

  def _gen_trajectory(self, n_traj, n_steps):
    # if it is not for visualization, we try our best to sample the trajectories
    # uniformly wihtin the whole region.
    dx_list = self.dx_list

    dx_idx = np.random.choice(len(dx_list), size=[n_traj * 10, n_steps])
    dx = dx_list[dx_idx]  # [N, T, 2]
    dx_cumsum = np.cumsum(dx, axis=1)  # [N, T, 2]

    # sample starting positions of trajectories
    x_start_max = np.fmin(
        self.num_grid - 3, np.min(self.num_grid - 2 - dx_cumsum, axis=1))
    x_start_min = np.fmax(3, np.max(-dx_cumsum + 2, axis=1))  # [N, 2]

    # we choose those traj where x_start_max > x_start_min
    select_idx = np.where(np.sum(x_start_max <= x_start_min, axis=1) == 0)[0]
    assert len(select_idx) >= n_traj
    select_idx = select_idx[:n_traj]
    x_start_max, x_start_min = x_start_max[select_idx], x_start_min[select_idx]
    dx_cumsum = dx_cumsum[select_idx]
    x_start = np.random.random((n_traj, 2)) * \
        (x_start_max - x_start_min) + x_start_min
    x_start = x_start[:, None]  # [N, 1, 2]
    x_start = np.round(x_start - 0.5)

    traj = np.concatenate((x_start, x_start + dx_cumsum), axis=1)

    return {'traj': traj}

  def _generate_dx_list(self, max_dr, interval=1.):
    dx_list = []
    max_dx_int = int(np.ceil(max_dr) + 1)
    for i in np.arange(0, max_dx_int, interval):
      for j in np.arange(0, max_dx_int, interval):
        if np.sqrt(i ** 2 + j ** 2) <= max_dr:
          dx_list.append(np.array([i, j]))
          if i > 0:
            dx_list.append(np.array([-i, j]))
          if j > 0:
            dx_list.append(np.array([i, -j]))
          if i > 0 and j > 0:
            dx_list.append(np.array([-i, -j]))
    dx_list = np.stack(dx_list)
    dx_list = dx_list.astype(np.float32)

    return dx_list
