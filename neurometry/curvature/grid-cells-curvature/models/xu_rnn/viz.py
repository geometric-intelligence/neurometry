import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import ImageSequenceClip

from neurometry.datasets.load_rnn_grid_cells import plot_rate_map

logs_dir = "logs/rnn_isometry"

activations_dir = "ckpt/activations"

def load_matrices(file_path):
    with open(file_path, "rb") as file:
        data = pickle.load(file)
    return data["u"], data["v"]


# def save_matrix_as_image(matrix, image_path):
#     plt.imshow(matrix, cmap='viridis')
#     plt.colorbar()
#     plt.savefig(image_path)
#     plt.close()


def create_video(image_folder, output_file, fps=10):
    image_files = [os.path.join(image_folder, img) for img in sorted(os.listdir(image_folder))]
    clip = ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(output_file, codec="libx264")


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


def save_rate_maps_as_image(indices, num_plots, activations, image_path, title, seed=None):
    plot_rate_map(indices, num_plots, activations, title, seed=seed)
    plt.savefig(image_path)
    plt.close()


def generate_videos(run_id, start_epoch=25000, end_epoch=65000, step=500, num_cells_per_image=20, seed=None):

    data_dir = os.path.join(logs_dir, run_id, activations_dir)
    #os.makedirs(output_dir, exist_ok=True)
    output_dir = data_dir
    u_images_dir = os.path.join(output_dir, "u_images")
    v_images_dir = os.path.join(output_dir, "v_images")
    os.makedirs(u_images_dir, exist_ok=True)
    os.makedirs(v_images_dir, exist_ok=True)

    rng = np.random.default_rng(seed=seed)
    idxs = rng.integers(0, 1799, num_cells_per_image)

    # Iterate through epochs
    for epoch in range(start_epoch, end_epoch, step):
        file_name = f"activations-step{epoch}.pkl"
        file_path = os.path.join(data_dir, file_name)
        u, v = load_matrices(file_path)

        # Save images using your custom plotting function
        u_image_path = os.path.join(u_images_dir, f"u_{epoch}.png")
        v_image_path = os.path.join(v_images_dir, f"v_{epoch}.png")
        save_rate_maps_as_image(idxs, num_cells_per_image, u, u_image_path, f"U matrices at epoch {epoch}", seed=seed)
        save_rate_maps_as_image(idxs, num_cells_per_image, v, v_image_path, f"V matrices at epoch {epoch}", seed=seed)

    # Create videos
    create_video(u_images_dir, os.path.join(output_dir, "u_video.mp4"))
    create_video(v_images_dir, os.path.join(output_dir, "v_video.mp4"))





