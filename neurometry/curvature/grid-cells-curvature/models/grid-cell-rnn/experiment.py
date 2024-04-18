""" Main training loop. """
import os

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from scores import GridScorer
from source import *

import ml_collections
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn

import input_pipeline
import model as model
import utils
import pickle

class Experiment:
  def __init__(self, config: ml_collections.ConfigDict, device):
    self.config = config
    self.device = device

    # initialize models
    logging.info("==== initialize model ====")
    self.model_config = model.GridCellConfig(**config.model)
    self.model = model.GridCell(self.model_config).to(device)

    # initialize dataset
    logging.info("==== initialize dataset ====")
    self.train_dataset = input_pipeline.TrainDataset(config.data, self.model_config)
    self.train_iter = iter(self.train_dataset)
    eval_dataset = input_pipeline.EvalDataset(
        config.integration, config.data.max_dr_trans, config.model.num_grid)
    self.eval_iter = iter(eval_dataset)

    # initialize optimizer
    logging.info("==== initialize optimizer ====")
    if config.train.optimizer_type=='adam': 
      self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.train.lr)
    elif config.train.optimizer_type=='adam_w': 
      self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.train.lr)
    elif config.train.optimizer_type=='sgd': 
      self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config.train.lr, momentum=0.9)

  def train_and_evaluate(self, workdir):
    logging.info('==== Experiment.train_and_evaluate() ===')

    if not tf.io.gfile.exists(workdir):
      tf.io.gfile.mkdir(workdir)
    config = self.config.train
    logging.info('num_steps_train=%d', config.num_steps_train)

    writer = metric_writers.create_default_writer(workdir)

    hooks = []
    report_progress = periodic_actions.ReportProgress(
        num_train_steps=config.num_steps_train, writer=writer)
    hooks += [report_progress]

    train_metrics = []
    block_size = self.model_config.block_size
    num_grid = self.model_config.num_grid
    num_block = self.model_config.num_neurons // block_size

    logging.info('==== Start of training ====')
    with metric_writers.ensure_flushes(writer):
      for step in range(1, config.num_steps_train+1):
        batch_data = utils.dict_to_device(next(self.train_iter), self.device)

        if 120000 > step > 10000: 
          lr = 0.0003
        elif step < 2000: # warm up
          lr = config.lr / 2000 * step + 3e-6
        elif step > 120000:
          lr = 0.0003 - (step - 120000) * (0.0003 / (config.num_steps_train - 120000))
        else: 
          lr = config.lr - (config.lr - 0.0003) / 10000 * step
        for param_group in self.optimizer.param_groups: 
          param_group['lr'] = lr
        
        self.optimizer.zero_grad()
        loss, metrics_step = self.model(batch_data, step)
        loss.backward()
        torch.nn.utils.clip_grad_norm(parameters=self.model.parameters(), max_norm=10)
        self.optimizer.step()
        
        if self.model_config.trans_type == 'nonlinear_simple': 
          if config.positive_v:
            with torch.no_grad():
              self.model.encoder.v.data = self.model.encoder.v.data.clamp(min=0.)
          
          # positive b
          self.model.trans.b.data = self.model.trans.b.data.abs()
        elif self.model_config.trans_type == 'lstm':
          if config.positive_u: 
            with torch.no_grad():
              self.model.decoder.u.data = self.model.decoder.u.data.clamp(min=0.)

        if config.norm_v:
          with torch.no_grad():
            v = self.model.encoder.v.data.reshape((-1, block_size, num_grid, num_grid))
            v_normed = nn.functional.normalize(v, dim=1) / np.sqrt(num_block)
            self.model.encoder.v.data = v_normed.reshape(
                (-1, num_grid, num_grid))

        metrics_step = utils.dict_to_numpy(metrics_step)
        train_metrics.append(metrics_step)

        # Quick indication that training is happening.
        logging.log_first_n(
            logging.WARNING, 'Finished training step %d.', 3, step)
        for h in hooks:
          h(step)

        if step % config.steps_per_logging == 0 or step == 1:
          train_metrics = utils.average_appended_metrics(train_metrics)
          writer.write_scalars(step, train_metrics)
          train_metrics = []

        if step % config.steps_per_large_logging == 0:
          # visualize v, u and heatmaps.
          with torch.no_grad():
            def visualize(weights, name):
              weights = weights.data.cpu().detach().numpy()
              weights = weights.reshape(
                  (-1, block_size, num_grid, num_grid))[:10, :10]
              writer.write_images(step, {name: utils.draw_heatmap(weights)})

            visualize(self.model.encoder.v, 'v')
            visualize(self.model.decoder.u, 'u')

            x_eval = torch.rand((3, 2)) * num_grid - 0.5
            x_eval = x_eval.to(self.device)
            v_x_eval = self.model.encoder(x_eval)
            x_pred, heatmaps, _ = self.model.decoder.decode(v_x_eval)
            
            # add fixed point condidtion check
            x1 = torch.arange(0,40,1).repeat_interleave(40)
            x2 = torch.arange(0,40,1).repeat(40)
            x1 = torch.unsqueeze(x1, 1)
            x2 = torch.unsqueeze(x2, 1)
            x = torch.cat((x1, x2), axis=1)

            error_fixed = 0.
            error_fixed_zero = 0.
            loss = nn.MSELoss()

            for i in range(40): 
                start = i * 40
                end = start + 40
                input = x[start:end, ]
                v_x = self.model.encoder(input.to(self.device))
                if self.config.model.trans_type == 'nonlinear_simple': 
                  trans_v_x = self.model.trans(v_x, torch.zeros((40, 2)).to(self.device))
                elif self.config.model.trans_type == 'lstm': 
                  trans_v_x = self.model.trans(v_x, torch.zeros((40, 1, 2)).to(self.device))
                x_t, _, _ = self.model.decoder.decode(trans_v_x)
                x_t_zero, _, _ = self.model.decoder.decode(v_x)
                error_fixed += loss(input.float().to(self.device), x_t.float())
                error_fixed_zero += loss(input.float().to(self.device), x_t_zero.float())

            error_fixed =  error_fixed / 40
            error_fixed_zero =  error_fixed_zero / 40
            
            heatmaps = heatmaps.cpu().detach().numpy()[None, ...]
            writer.write_images(
                step, {'vu_heatmap': utils.draw_heatmap(heatmaps)})

            err = torch.mean(torch.sum((x_eval - x_pred) ** 2, dim=-1))
            writer.write_scalars(step, {'pred_x': err.item()})
            writer.write_scalars(step, {'error_fixed': error_fixed.item()})
            writer.write_scalars(step, {'error_fixed_zero': error_fixed_zero.item()})

        if step % config.steps_per_integration == 0 or step == 1:
          # perform path integration
          with torch.no_grad():
            eval_data = utils.dict_to_device(next(self.eval_iter), self.device)

            if 10000 < step < 20000: 
              scale_tensor, score, max_scale = self.grid_scale()
              scaling = max_scale * num_grid / self.config.data.max_dr_isometry
              scale_tensor = scale_tensor / scaling
              self.train_dataset.scale_vector = (scale_tensor * num_grid).detach().numpy()
              print((scale_tensor * num_grid).detach().numpy())

              writer.write_scalars(step, {'score': score.item()})
              writer.write_scalars(step, {'scale': scale_tensor[0].item() * num_grid})
              writer.write_scalars(step, {'scale_mean': torch.mean(scale_tensor).item() * num_grid})

            # for visualization
            if self.config.model.trans_type == 'nonlinear_simple': 
              outputs = self.model.path_integration(**eval_data['traj_vis'])
            elif self.config.model.trans_type == 'lstm': 
              outputs = self.model.path_integration_lstm(**eval_data['traj_vis'])
              
            outputs = utils.dict_to_numpy(outputs)
            images = {
                # [N, T, 2]
                'trajs': utils.draw_trajs(outputs['traj_real'], outputs['traj_pred']['vanilla'], num_grid),
                'trajs_reencode': utils.draw_trajs(outputs['traj_real'], outputs['traj_pred']['reencode'], num_grid),
                # [N, T[::5], H, W]
                'heatmaps': utils.draw_heatmap(outputs['heatmaps'][:, ::5]),
            }
            writer.write_images(step, images)

            # for quantitative evaluation
            if self.config.model.trans_type == 'nonlinear_simple': 
              outputs = self.model.path_integration(**eval_data['traj'])
            elif self.config.model.trans_type == 'lstm': 
              outputs = self.model.path_integration_lstm(**eval_data['traj'])

            err = utils.dict_to_numpy(outputs['err'])
            writer.write_scalars(step, err)

        if step == config.num_steps_train:
          ckpt_dir = os.path.join(workdir, 'ckpt')
          if not tf.io.gfile.exists(ckpt_dir):
            tf.io.gfile.makedirs(ckpt_dir)
          self._save_checkpoint(step, ckpt_dir)

  def grid_scale(self): 
    num_interval = self.model_config.num_grid
    block_size = self.model_config.block_size
    num_block = self.model_config.num_neurons // self.model_config.block_size

    starts = [0.1] * 20
    ends = np.linspace(0.2, 1.4, num=20)

    masks_parameters = zip(starts, ends.tolist())

    ncol, nrow = block_size, num_block
    weights = self.model.encoder.v.data.cpu().detach().numpy()

    scorer = GridScorer(40, ((0, 1), (0, 1)), masks_parameters)

    score_list = np.zeros(shape=[len(weights)], dtype=np.float32)
    scale_list = np.zeros(shape=[len(weights)], dtype=np.float32)
    orientation_list = np.zeros(shape=[len(weights)], dtype=np.float32)
    sac_list = []
    # plt.figure(figsize=(int(ncol * 1.6), int(nrow * 1.6)))

    for i in range(len(weights)):
      rate_map = weights[i]
      rate_map = (rate_map - rate_map.min()) / (rate_map.max() - rate_map.min())
      '''
      score, autocorr_ori, autocorr, scale, orientation, peaks = \
          gridnessScore(rateMap=rate_map, arenaDiam=1, h=1.0 /
                        (num_interval-1), corr_cutRmin=0.3)
                        
      if (i > 64 and i < 74) or (i > 74 and i < 77) or (i > 77 and i < 89) or (i > 89 and i < 92) or (i > 92 and i < 96):
        peaks = peaks0
      else:
        peaks0 = peaks
      '''
      score_60, score_90, max_60_mask, max_90_mask, sac = scorer.get_scores(
          weights[i])
      sac_list.append(sac)

      score_list[i] = score_60
      # scale_list[i] = scale
      scale_list[i] = max_60_mask[1]
      # orientation_list[i] = orientation

    scale_tensor = torch.from_numpy(scale_list)
    score_tensor = torch.from_numpy(score_list)
    max_scale = torch.max(scale_tensor[score_list > 0.37])

    scale_tensor = scale_tensor.reshape((num_block, block_size))
    scale_tensor = torch.mean(scale_tensor, dim=1)
    
    # score_tensor = score_tensor.reshape((num_block, block_size))
    score_tensor = torch.mean(score_tensor)

    return scale_tensor, score_tensor, max_scale

  def _save_checkpoint(self, step, ckpt_dir):
    """
    Saving checkpoints
    :param epoch: current epoch number
    :param log: logging information of the epoch
    :param save_best: if True, rename the saved checkpoint to 'model_best.pth'
    """
    arch = type(self.model).__name__
    state = {
        'arch': arch,
        'step': step,
        'state_dict': self.model.state_dict(),
        'optimizer': self.optimizer.state_dict(),
        'config': self.config
    }
    filename = os.path.join(ckpt_dir, 'checkpoint-step{}.pth'.format(step))
    torch.save(state, filename)
    logging.info("Saving checkpoint: {} ...".format(filename))
