import numpy as np



def get_lattice(scale, lattice_type, dimensions):
    
    if lattice_type == "square":
        lx = scale
        ly = scale
    elif lattice_type == "hexagonal":
        lx = scale
        ly = scale * np.sqrt(3) / 2
    else:
        raise NotImplementedError 
        
    
    n_x = np.arange(-(dimensions[0] / lx), (dimensions[0] / lx) + 1)
    n_y = np.arange(-(dimensions[1] / ly), (dimensions[1] / ly) + 1)
    N_x, N_y = np.meshgrid(n_x, n_y)
    
    if lattice_type == "hexagonal":
        offset_x = np.tile([[0], [0.5]], np.shape(N_x))[: np.shape(N_x)[0], :]
        X = lx * (N_x - offset_x)
        Y = ly * N_y
    elif lattice_type == "square":
        X = lx*N_x
        Y = ly * N_y
        
    lattice = np.hstack((np.reshape(X, (-1, 1)), np.reshape(Y, (-1, 1))))
    
    return lattice