import ml_collections
import torch


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    """Get the hyperparameters for the model"""
    config = ml_collections.ConfigDict()
    config.gpu = 3

    # training config
    config.train = d(
        load_pretrain=False,
        pretrain_dir="logs/rnn_isometry/20240418-180712/ckpt/model/checkpoint-step25000.pth",
        num_steps_train=30000,  # 100000
        lr=0.006,
        lr_decay_from=10000,
        steps_per_logging=20,
        steps_per_large_logging=500,  # 500
        steps_per_integration=2000,
        norm_v=True,
        positive_v=True,
        positive_u=False,
        optimizer_type="adam",
    )

    # simulated data
    config.data = d(
        max_dr_trans=3.0,
        max_dr_isometry=15.0,
        batch_size=10000,
        sigma_data=0.48,
        add_dx_0=False,
        small_int=False,
    )

    # model parameter
    config.model = d(
        trans_type="nonlinear_simple",
        rnn_step=200, #10
        num_grid=40,
        num_neurons=1800,
        block_size=12,
        sigma=0.07,
        w_kernel=1.05,
        w_trans=0.5, #0.1
        w_isometry=0.005,
        w_reg_u=0.2,
        reg_decay_until=15000,
        adaptive_dr=True,
        s_0 = 1000,
        x_saliency = 0.8,
        sigma_saliency = 0.1,
        reward_step = 10000,
        saliency_type = "gaussian",
    )

    # path integration
    config.integration = d(
        n_inte_step=50, # 50
        n_traj=100,
        n_inte_step_vis=50,
        n_traj_vis=5,
    )

    return config
