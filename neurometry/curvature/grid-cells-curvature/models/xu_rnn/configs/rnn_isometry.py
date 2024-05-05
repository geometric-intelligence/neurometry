import ml_collections
import torch


def d(**kwargs):
    """Helper of creating a config dict."""
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def get_config():
    """Get the hyperparameters for the model"""
    config = ml_collections.ConfigDict()
    config.gpu = 5

    # training config
    config.train = d(
        load_pretrain=True,
        pretrain_dir="logs/rnn_isometry/20240418-180712/ckpt/model/checkpoint-step25000.pth",
        num_steps_train=40000,  # 100000
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
        rnn_step=10,
        num_grid=40,
        num_neurons=1800,
        block_size=12,
        sigma=0.07,
        w_kernel=1.05,
        w_trans=10,#0.1
        w_isometry=0.005,
        w_reg_u=0.2,
        reg_decay_until=15000,
        adaptive_dr=True,
        s_0 = 5,
        x_star = torch.tensor([0.5, 0.5]),
        sigma_star_x = 0.2,
        sigma_star_y = 0.2,
        reward_step = 3000,
        saliency_type = "left_half",
    )

    # path integration
    config.integration = d(
        n_inte_step=30, # 50
        n_traj=100,
        n_inte_step_vis=30,
        n_traj_vis=5,
    )

    return config
