""" Main function for training the representation model of grid cells. """
import os

from absl import app
from absl import flags
from ml_collections import config_flags
import numpy as np
import tensorflow as tf

import experiment
import utils

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", "../logs", "Work unit directory.")
flags.mark_flags_as_required(["config"])
flags.DEFINE_string("mode", 'train', "train / visualize / integration / correction")


def main(argv):
  del argv
  config = FLAGS.config

  np.random.seed(0)

  # config workdir
  if FLAGS.mode == "train":
    workdir = os.path.join(FLAGS.workdir, utils.get_workdir())
  else:
    workdir = os.path.join(FLAGS.workdir, "eval")
  if not tf.io.gfile.exists(workdir):
    tf.io.gfile.makedirs(workdir)

  # record config file
  with open(os.path.join(workdir, "config.txt"), 'a') as f:
    print(config, file=f)

  device = utils.get_device(config.gpu)

  if FLAGS.mode == "train":  # training
    exp = experiment.Experiment(config, device)
    exp.train_and_evaluate(workdir)


if __name__ == '__main__':
  app.run(main)
