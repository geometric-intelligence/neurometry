import os

import numpy as np
import torch
from visualize import save_ratemaps


class Trainer:
    def __init__(self, options, model, trajectory_generator, restore=False):
        self.options = options
        self.model = model
        self.trajectory_generator = trajectory_generator
        lr = self.options.learning_rate
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss = []
        self.err = []

        # Set up checkpoints
        self.ckpt_dir = os.path.join(options.save_dir, options.run_ID)
        ckpt_path = os.path.join(self.ckpt_dir, "most_recent_model.pth")
        if restore and os.path.isdir(self.ckpt_dir) and os.path.isfile(ckpt_path):
            self.model.load_state_dict(torch.load(ckpt_path))
            print(f"Restored trained model from {ckpt_path}")
        else:
            if not os.path.isdir(self.ckpt_dir):
                os.makedirs(self.ckpt_dir, exist_ok=True)
            print("Initializing new model from scratch.")
            print(f"Saving to: {self.ckpt_dir}")

    def train_step(self, inputs, pc_outputs, pos):
        """
        Train on one batch of trajectories.

        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        """
        self.model.zero_grad()

        loss, err = self.model.compute_loss(inputs, pc_outputs, pos)

        loss.backward()
        self.optimizer.step()

        return loss.item(), err.item()

    def train(self, n_epochs: int = 1000, n_steps=10, save=True):
        """
        Train model on simulated trajectories.

        Args:
            n_steps: Number of training steps
            save: If true, save a checkpoint after each epoch.
        """

        # Construct generator
        gen = self.trajectory_generator.get_generator()

        # tbar = tqdm(range(n_steps), leave=False)
        for epoch_idx in range(n_epochs):
            for step_idx in range(n_steps):
                inputs, pc_outputs, pos = next(gen)
                loss, err = self.train_step(inputs, pc_outputs, pos)
                self.loss.append(loss)
                self.err.append(err)

                # Log error rate to progress bar
                # tbar.set_description('Error = ' + str(np.int(100*err)) + 'cm')

                print(
                    f"Epoch: {epoch_idx}/{n_epochs}. Step {step_idx}/{n_steps}. Loss: {np.round(loss, 2)}. Err: {np.round(100 * err, 2)}cm"
                )

            if save and (epoch_idx % 5) == 0:
                # Save checkpoint
                ckpt_path = os.path.join(
                    self.ckpt_dir, f"epoch_{epoch_idx}.pth"
                )
                torch.save(self.model.state_dict(), ckpt_path)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(self.ckpt_dir, "most_recent_model.pth"),
                )

                # Save a picture of rate maps
                save_ratemaps(
                    self.model, self.trajectory_generator, self.options, step=epoch_idx
                )
