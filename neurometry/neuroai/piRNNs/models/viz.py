import os
import pickle

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip

logs_dir = "logs/rnn_isometry"

activations_dir = "ckpt/activations"

def load_matrices(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data["u"], data["v"]


def create_video(image_folder, output_file, fps=10):
    image_files = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder))]
    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(output_file, codec="libx264")



def draw_heatmap(activations, title):
    # activations should a 4-D tensor: [M, N, H, W]
    nrow, ncol = activations.shape[0], activations.shape[1]
    fig = plt.figure(figsize=(ncol, nrow))

    for i in range(nrow):
        for j in range(ncol):
            plt.subplot(nrow, ncol, i * ncol + j + 1)
            weight = activations[i, j]
            vmin, vmax = weight.min() - 0.01, weight.max()

            cmap = cm.get_cmap("jet", 1000)
            cmap.set_under("w")

            plt.imshow(
                weight,
                interpolation="nearest",
                cmap=cmap,
                aspect="auto",
                vmin=vmin,
                vmax=vmax,
            )
            plt.axis("off")

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    fig.suptitle(title, fontsize=20, fontweight="bold", verticalalignment="top")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    #plt.close(fig)

    return fig


def save_rate_maps_as_image(activations, image_path, title):
    fig = draw_heatmap(activations, title)
    plt.savefig(image_path)
    plt.close()
    return fig


def generate_videos(run_id, start_epoch=25000, end_epoch=65000, step=500):

    data_dir = os.path.join(logs_dir, run_id, activations_dir)
    #os.makedirs(output_dir, exist_ok=True)
    output_dir = data_dir
    u_images_dir = os.path.join(output_dir, "u_images")
    v_images_dir = os.path.join(output_dir, "v_images")
    os.makedirs(u_images_dir, exist_ok=True)
    os.makedirs(v_images_dir, exist_ok=True)

    #config_file = os.path.join(logs_dir, run_id, "config.txt")
    # with open(config_file, 'r') as file:
    #     config = yaml.safe_load(file)

    # block_size = config["model"]["block_size"]
    # num_grid = config["model"]["num_grid"]
    block_size = 12
    num_grid = 40


    for epoch in range(start_epoch, end_epoch, step):
        file_name = f"activations-step{epoch}.pkl"
        file_path = os.path.join(data_dir, file_name)
        u, v = load_matrices(file_path)
        u = u.reshape(-1,block_size,num_grid,num_grid)[:10,:10]
        v = v.reshape(-1,block_size,num_grid,num_grid)[:10,:10]

        u_image_path = os.path.join(u_images_dir, f"Q_{epoch}.png")
        v_image_path = os.path.join(v_images_dir, f"rate_map_{epoch}.png")
        save_rate_maps_as_image(u, u_image_path, f"Q matrices at epoch {epoch}")
        save_rate_maps_as_image(v, v_image_path, f"Grid cell rate maps at epoch {epoch}")

    # Create videos
    create_video(u_images_dir, os.path.join(output_dir, "Q_video.mp4"))
    create_video(v_images_dir, os.path.join(output_dir, "rate_map_video.mp4"))



# def save_rate_maps_as_image(indices, num_plots, activations, image_path, title, seed=None):
#     plot_rate_map(indices, num_plots, activations, title, seed=seed)
#     plt.savefig(image_path)
#     plt.close()


# def generate_videos(run_id, start_epoch=25000, end_epoch=65000, step=500):

#     data_dir = os.path.join(logs_dir, run_id, activations_dir)

#     print(data_dir)

#     output_dir = data_dir
#     os.makedirs(output_dir, exist_ok=True)
#     u_images_dir = os.path.join(output_dir, 'u_images')
#     v_images_dir = os.path.join(output_dir, 'v_images')
#     os.makedirs(u_images_dir, exist_ok=True)
#     os.makedirs(v_images_dir, exist_ok=True)

#     # Generate images
#     for epoch in range(start_epoch, end_epoch, step):
#         file_name = f'activations-step{epoch}.pkl'
#         file_path = os.path.join(data_dir, file_name)
#         u, v = load_matrices(file_path)
#         save_matrix_as_image(u, os.path.join(u_images_dir, f'u_{epoch}.png'))
#         save_matrix_as_image(v, os.path.join(v_images_dir, f'v_{epoch}.png'))

#     # Create videos
#     create_video(u_images_dir, os.path.join(output_dir, 'u_video.mp4'))
#     create_video(v_images_dir, os.path.join(output_dir, 'v_video.mp4'))





