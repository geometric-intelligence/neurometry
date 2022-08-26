import torch
from abn.trainer.core import Trainer


class SphericalVAETrainer(Trainer):
    def __init__(
        self,
        model,
        optimizer,
        beta=0.1,
        logger=None,
        scheduler=None,
    ):
        """
        beta is a coefficient on the kl loss
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            logger=logger,
            scheduler=scheduler,
            loss=None,
            regularizer=None,
            normalizer=None,
        )

        self.beta = beta

    def step(self, data_loader, grad=True):

        # log_dict keeps track of average losses per epoch
        log_dict = {
            "recon_loss": 0,
            "kl_loss": 0,
            "total_loss": 0,
            "elbo": 0,
            "log_likelihood": 0,
        }

        # Dataset passes pairs of neural state vectors and the ground truth positional angle
        # between the corresponding timepoints. However, we are not using this info in this model
        for i, (x, x1, angle) in enumerate(data_loader):

            x = x.to(self.model.device)

            if grad:
                self.optimizer.zero_grad()
                (z_mean, z_var), (q_z, p_z), z, x_ = self.model.forward(x)

            else:
                with torch.no_grad():
                    (z_mean, z_var), (q_z, p_z), z, x_ = self.model.forward(x)

            recon_loss = (
                torch.nn.BCEWithLogitsLoss(reduction="none")(x_, x).sum(-1).mean()
            )

            if self.model.distribution == "normal":
                kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif self.model.distribution == "vmf":
                kl_loss = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            else:
                raise NotImplemented

            total_loss = recon_loss + self.beta * kl_loss

            if grad:
                total_loss.backward()
                self.optimizer.step()

            log_dict["recon_loss"] += recon_loss
            log_dict["kl_loss"] += kl_loss
            log_dict["total_loss"] += total_loss
            log_dict["elbo"] += -recon_loss - kl_loss
            log_dict["log_likelihood"] += self.model.log_likelihood(x)

        # Normalize loss terms for the number of samples/batches in the epoch
        n_samples = len(data_loader)
        for key in log_dict.keys():
            log_dict[key] /= n_samples

        plot_variable_dict = {"model": self.model}

        return log_dict, plot_variable_dict
