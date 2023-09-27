import numpy as np

def get_voxels_coordinates(voxels_list):
    coordinates = []

    for coord_str in voxels_list:
        split_str = coord_str.split("-")
        x = int(split_str[1])
        y = int(split_str[2])
        z = int(split_str[3])
        coordinates.append((x, y, z))

    coordinates = np.array(coordinates)

    centroid = np.mean(coordinates, axis=0)

    return coordinates, centroid