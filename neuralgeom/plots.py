import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import numpy as np



def create_plots(train_losses,test_losses,model,dataset_torch,labels,angles,mean_curvature_norms,mean_curvature_norms_analytic, error, config):
    fig_loss = plot_loss(train_losses, test_losses, config)
    fig_recon = plot_recon(model, dataset_torch, labels, angles, config)
    fig_latent = plot_latent_space(model, dataset_torch, labels, config)
    if config.dataset_name == "s1_synthetic":
        fig_curv, fig_curv_analytic = plot_curv(angles, mean_curvature_norms, mean_curvature_norms_analytic, config)
        fig_comparison = plot_comparison(angles, mean_curvature_norms_analytic, mean_curvature_norms, error, config)
    else:
        fig_comparison = None
    if config.dataset_name == "s2_synthetic":
        fig_curv, fig_curv_analytic, fig_comparison = [None for _ in range(3)]

    return fig_loss, fig_recon, fig_latent, fig_curv, fig_curv_analytic, fig_comparison



def plot_loss(train_losses, test_losses, config):
    fig, ax = plt.subplots(figsize=(20,20))
    epochs = [epoch for epoch in range(1,config.n_epochs+1)]
    ax.plot(epochs,train_losses, label="train")
    ax.plot(epochs,test_losses, label="test")
    ax.set_title("Losses", fontsize=40)
    ax.set_xlabel("epoch", fontsize=40)
    ax.legend(prop={"size": 40})
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.savefig(f"results/figures/{config.results_prefix}_losses.png")
    plt.savefig(f"results/figures/{config.results_prefix}_losses.svg")
    return fig


def plot_recon(model, dataset_torch, labels, angles, config):
    fig = plt.figure(figsize=(24,12))

    if config.dataset_name == "s1_synthetic":
        ax_data = fig.add_subplot(1,2,1)
        colormap = plt.get_cmap("hsv")
        x_data = dataset_torch[:, 0]
        y_data = dataset_torch[:, 1]
        z = torch.stack([torch.cos(angles), torch.sin(angles)], axis=-1)
        z = z.to(config.device)
        rec = model.decode(z)
        x_rec = rec[:, 0]
        x_rec = [x.item() for x in x_rec]
        y_rec = rec[:, 1]
        y_rec = [y.item() for y in y_rec]
        ax_data.set_title("Synthetic data", fontsize=40)
        sc_data = ax_data.scatter(x_data, y_data, c=labels["angles"], cmap=colormap)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        ax_rec = fig.add_subplot(1,2,2)
        ax_rec.set_title("Reconstruction", fontsize=40)
        sc_rec = ax_rec.scatter(x_rec, y_rec, c=labels["angles"], cmap=colormap)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        #plt.colorbar(sc_rec)
    elif config.dataset_name == "s2_synthetic":
        ax_data = fig.add_subplot(1,2,1, projection="3d")
        x_data = dataset_torch[:,0]
        y_data = dataset_torch[:,1]
        z_data = dataset_torch[:,2]
        norms_data = torch.linalg.norm(dataset_torch,axis=1).detach().numpy()
        thetas = angles[:,0]
        phis = angles[:,1]
        z = torch.stack([torch.sin(thetas)*torch.cos(phis), torch.sin(thetas)*torch.sin(phis), torch.cos(thetas)], axis=-1)
        z = z.to(config.device)
        rec = model.decode(z)
        norms_rec = torch.linalg.norm(rec,axis=1).detach().numpy()
        x_rec = rec[:,0]
        x_rec = [x.item() for x in x_rec]
        y_rec = rec[:,1]
        y_rec = [y.item() for y in y_rec]
        z_rec = rec[:,2]
        z_rec = [w.item() for w in z_rec]

        ax_data.set_title("Synthetic data", fontsize=40)
        sc_data = ax_data.scatter3D(x_data, y_data, z_data, s=5, c = norms_data)
        #ax_data.view_init(elev=60, azim=45, roll=0)
        ax_rec = fig.add_subplot(1,2,2, projection="3d")
        ax_rec.set_title("Reconstruction", fontsize=40)
        sc_rec = ax_rec.scatter3D(x_rec,y_rec,z_rec,s=5,c = norms_rec)
    elif config.dataset_name == "experimental":
        z = torch.stack([torch.cos(angles), torch.sin(angles)], axis=-1)
        image_data = ax_data.imshow(dataset_torch, aspect=0.05)
        z = torch.stack([torch.cos(angles), torch.sin(angles)], axis=-1)
        z = z.to(config.device)
        rec = model.decode(z).detach().cpu().numpy()
        ax_rec.set_title("Reconstruction", fontsize=40)
        image_rec = ax_rec.imshow(rec, aspect= 0.05)
        plt.colorbar(image_rec)
    else:
        raise NotImplementedError

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.savefig(f"results/figures/{config.results_prefix}_recon.png")
    plt.savefig(f"results/figures/{config.results_prefix}_recon.svg")
    return fig


def plot_latent_space(model, dataset_torch, labels, config):
    fig = plt.figure(figsize=(20,20))
    if config.dataset_name in ("s1_synthetic", "experimental"):
        ax = fig.add_subplot(111)
        _, posterior_params = model(dataset_torch.to(config.device))
        z, _, _ = model.reparameterize(posterior_params)
        colormap = plt.get_cmap("twilight")
        z0 = z[:, 0]
        z0 = [_.item() for _ in z0]
        z1 = z[:, 1]
        z1 = [_.item() for _ in z1]
        sc = ax.scatter(z0, z1, c=labels["angles"], s=10, cmap=colormap)
        ax.set_xlim(-1.2,1.2)
        ax.set_ylim(-1.2,1.2)
    elif config.dataset_name == "s2_synthetic":
        ax = fig.add_subplot(111,projection="3d")
        _, posterior_params = model(dataset_torch.to(config.device))
        z, _, _ = model.reparameterize(posterior_params)
        z0 = z[:,0]
        z0 = [_.item() for _ in z0]
        z1 = z[:, 1]
        z1 = [_.item() for _ in z1]
        z2 = z[:, 2]
        z2 = [_.item() for _ in z2]
        sc = ax.scatter3D(z0,z1,z2)
        ax.view_init(elev=60, azim=45, roll=0)
        ax.set_xlim(-1.2,1.2)
        ax.set_ylim(-1.2,1.2)
        ax.set_zlim(-1.2,1.2)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    ax.set_title("Latent space", fontsize=40)
    plt.savefig(f"results/figures/{config.results_prefix}_latent_plot.png")
    plt.savefig(f"results/figures/{config.results_prefix}_latent_plot.svg")
    return fig 


def plot_curv(angles, mean_curvature_norms, mean_curvature_norms_analytic, config):

    if config.dataset_name == "s1_synthetic":
        colormap = plt.get_cmap("hsv")
        color_norm = mpl.colors.Normalize(0.0, 1.2 * max(mean_curvature_norms_analytic))
        fig_analytic= plt.figure(figsize=(40,20))
        ax1_analytic = plt.subplot(121, projection='polar')
        ax2_analytic = plt.subplot(122)
        sc_analytic = ax1_analytic.scatter(
            angles,
            np.ones_like(angles),
            c=mean_curvature_norms_analytic,
            s=20,
            cmap=colormap,
            norm=color_norm,
            linewidths=0,
        )
        ax1_analytic.set_yticks([])
    
        plt.xticks(fontsize=30)
        plt.colorbar(sc_analytic)
        ax2_analytic.plot(angles, mean_curvature_norms_analytic)
        ax1_analytic.set_title("Visualizing mean curvature on circle", fontsize=40)
        ax2_analytic.set_title("Analytic mean curvature profile", fontsize=40)
        ax2_analytic.set_xlabel("angle", fontsize=40)
        plt.savefig(f"results/figures/{config.results_prefix}_curv_profile_analytic.png")
        plt.savefig(f"results/figures/{config.results_prefix}_curv_profile_analytic.svg")
    else:
        fig_analytic = None

    fig = plt.figure(figsize=(40,20))
    ax1 = plt.subplot(121, projection='polar')
    ax2 = plt.subplot(122)

    if config.dataset_name in ("experimental", "s1_synthetic"):
        colormap = plt.get_cmap("hsv")
        color_norm = mpl.colors.Normalize(0.0, 1.2 * max(mean_curvature_norms))
        sc = ax1.scatter(
            angles,
            np.ones_like(angles),
            c=mean_curvature_norms,
            s=20,
            cmap=colormap,
            norm=color_norm,
            linewidths=0,
        )
        ax1.set_yticks([])
    
        plt.xticks(fontsize=30)
        plt.colorbar(sc)
        ax2.plot(angles, mean_curvature_norms)
        ax2.set_title("Learned mean curvature profile", fontsize=40)
        ax1.set_title("Visualizing mean curvature on circle", fontsize=40)
        ax2.set_xlabel("angle", fontsize=40)
        plt.savefig(f"results/figures/{config.results_prefix}_curv_profile_learned.png")
        plt.savefig(f"results/figures/{config.results_prefix}_curv_profile_learned.svg")
    
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    return fig, fig_analytic


def plot_comparison(angles, mean_curvature_norms_analytic, mean_curvature_norms, error, config):
    fig, ax = plt.subplots(figsize=(20,20))

    if config.dataset_name == "s1_synthetic":
        ax.plot(angles, mean_curvature_norms_analytic,"--",label="analytic")
        ax.plot(angles,mean_curvature_norms,label="learned")
        ax.set_xlabel("angle", fontsize=40)
    
    ax.legend(prop={"size": 40}, loc = "upper right")
    ax.set_title("Error = " + "%.3f" % error, fontsize=30)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.savefig(f"results/figures/{config.results_prefix}_comparison.png")
    plt.savefig(f"results/figures/{config.results_prefix}_comparison.svg")
    return fig







