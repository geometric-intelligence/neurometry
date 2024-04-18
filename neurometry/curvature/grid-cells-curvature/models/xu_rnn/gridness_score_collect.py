from scores import GridScorer
import numpy as np
from matplotlib import pyplot as plt
from source import *
from utils import draw_heatmap_2D
import os
import argparse
import pickle
import torch
import utils
import model

parser = argparse.ArgumentParser()
# 1-step RNN
# parser.add_argument('--f_in', type=str, default='../logs/01_isometry/20220727-223216-num_neurons=1800-005-1-positive_v=True-num_steps_train=200000-batch_size=10000-006-gpu=0/ckpt/weights.npy', help='Checkpoint path to load')
# parser.add_argument('--f_in', type=str, default='../logs/04_rnn_isometry/20220827-234250-rnn_step=1-batch_size=8000-gpu=0/ckpt/checkpoint-step100000.pth', help='Checkpoint path to load')

# 5-step RNN
# parser.add_argument('--f_in', type=str, default='../logs/04_rnn_isometry/20220828-165259-rnn_step=1-adaptive_dr=True-reg_decay_until=20000-batch_size=8000-gpu=0/ckpt/weights.npy', help='Checkpoint path to load')


# 10-step RNN
# parser.add_argument('--f_in', type=str, default='../logs/01_isometry_rnn/20220802-215231-num_steps_train=200000-gpu=1/ckpt/weights.npy', help='Checkpoint path to load')
parser.add_argument('--f_in', type=str, default='../logs/04_rnn_isometry/20220915-223938-rnn_step=10-block_size=12-005-1-adaptive_dr=True-reg_decay_until=15000-batch_size=8000-num_steps_train=100000-gpu=0/ckpt/checkpoint-step100000.pth', help='Checkpoint path to load')

# parser.add_argument('--f_in', type=str, default='/home/gmm/Documents/workingspace/grid_cell_00/output/main_100_00_new_loss_small_area/2021-05-24-17-49-02--num_group=1--block_size=96--num_data=20000--weight_reg_u=6/syn/weights_7999.npy', help='Checkpoint path to load')
parser.add_argument('--dir_out', type=str, default='test',
                    help='Checkpoint path to load')
FLAGS = parser.parse_args()

# read ckpt
ckpt_path = FLAGS.f_in
ckpt = torch.load(ckpt_path)

config = ckpt['config']

device = utils.get_device(1)
# config.b_scalar = True

model_config = model.GridCellConfig(**config.model)
model = model.GridCell(model_config)
model.load_state_dict(ckpt['state_dict'])
model.to(device)

# np.save('../logs/04_rnn_isometry/20220828-165259-rnn_step=1-adaptive_dr=True-reg_decay_until=20000-batch_size=8000-gpu=0/ckpt/weights.npy', \
#         model.encoder.v.data.cpu().numpy())

dir_out = './output/test_gridness'
log_file = os.path.join(dir_out, 'log.txt')

dir_out = os.path.join(dir_out, FLAGS.dir_out)
if not os.path.exists(dir_out):
  os.mkdir(dir_out)
num_interval = 40
block_size = 12
num_block = 150

starts = [0.1] * 20
ends = np.linspace(0.2, 1.2, num=20)

# starts = [0.2] * 10
# ends = np.linspace(0.4, 1.6, num=20)

# starts = [0.1] * 30 + [0.2] * 30
# ends = np.concatenate([np.linspace(0.2, 1.5, num=30), np.linspace(0.3, 1.5, num=30)])


masks_parameters = zip(starts, ends.tolist())

# weights_file = FLAGS.f_in
# weights = np.load(weights_file)
weights = model.encoder.v.data.cpu().numpy()
# weights = np.transpose(weights, axes=[2, 0, 1])
ncol, nrow = block_size, num_block

scorer = GridScorer(40, ((0, 1), (0, 1)), masks_parameters)

score_list = np.zeros(shape=[len(weights)], dtype=np.float32)
scale_list = np.zeros(shape=[len(weights)], dtype=np.float32)
orientation_list = np.zeros(shape=[len(weights)], dtype=np.float32)
sac_list = []
plt.figure(figsize=(int(ncol * 1.6), int(nrow * 1.6)))


for i in range(len(weights)):
  rate_map = weights[i]
  rate_map = (rate_map - rate_map.min()) / (rate_map.max() - rate_map.min())
  
  score, autocorr_ori, autocorr, scale, orientation, peaks = \
      gridnessScore(rateMap=rate_map, arenaDiam=1, h=1.0 /
                    (num_interval-1), corr_cutRmin=0.3)
                    
  if (i > 64 and i < 74) or (i > 74 and i < 77) or (i > 77 and i < 89) or (i > 89 and i < 92) or (i > 92 and i < 96):
    peaks = peaks0
  else:
    peaks0 = peaks
  

  score_60, score_90, max_60_mask, max_90_mask, sac = scorer.get_scores(
      weights[i])
  sac_list.append(sac)
  '''
  scorer.plot_sac(autocorr,
                  ax=plt.subplot(nrow, ncol, i + 1),
                  title="%.2f" % (score_60),
                  # title="%.2f, %.2f, %.2f" % (score_60, scale, orientation),
                  cmap='jet')
  '''
  
  scorer.plot_sac(sac,
                  ax=plt.subplot(nrow, ncol, i + 1),
                  title="",
                  # title="%.2f" % (score_60),
                  # title="%.2f, %.2f, %.2f" % (score_60, scale, orientation),
                  cmap='jet')
  '''
  scorer.plot_sac(sac,
                  ax=plt.subplot(nrow, ncol, i + 1),
                  title="%.2f" % (max_60_mask[1]),
                  # title="%.2f, %.2f, %.2f" % (score_60, scale, orientation),
                  cmap='jet')
                  '''
  plt.subplots_adjust(wspace=0.2, hspace=0.2)
  score_list[i] = score_60
  # scale_list[i] = scale
  # print(max_60_mask)
  scale_list[i] = max_60_mask[1]
  orientation_list[i] = orientation
# plt.savefig(os.path.join(dir_out, 'autocorr.png'), bbox_inches='tight')
plt.savefig(os.path.join(dir_out, 'autocorr_score_noscore.png'), bbox_inches='tight')
# plt.savefig(os.path.join(dir_out, 'polar.png'))
plt.close()
sac_list = np.asarray(sac_list)

# with open(os.path.join(dir_out, 'stats.pkl'), "wb") as f:
#   pickle.dump([sac_list, score_list, scale_list, orientation_list], f)
# np.set_printoptions(threshold=np.nan)
np.save(os.path.join(dir_out, 'score_list.npy'), score_list)
np.save(os.path.join(dir_out, 'scale_list.npy'), scale_list)
np.save(os.path.join(dir_out, 'orientation_list.npy'), orientation_list)

scale_list = np.load(os.path.join(dir_out, 'scale_list.npy'))
score_list = np.load(os.path.join(dir_out, 'score_list.npy'))
orientation_list = np.load(os.path.join(dir_out, 'orientation_list.npy'))

print(score_list)
print(len(score_list[np.isnan(score_list)]))
print(np.mean(score_list[~np.isnan(score_list)]))

print(np.mean(scale_list))
print(len(scale_list))
print((scale_list * 40))
print(np.sum(score_list > 0.37) / len(score_list))

plt.hist(orientation_list, density=True, bins=20)
plt.show()
plt.hist(orientation_list[score_list > 0.37], density=True, bins=20)
plt.show()
# with open(os.path.join(dir_out, 'stats.pkl'), "rb") as f:
#   sac_list, score_list, scale_list, orientation_list = pickle.load(f)
