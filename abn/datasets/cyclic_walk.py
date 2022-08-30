import torch
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm
import datasets.utils as utils
import matplotlib.pyplot as plt
import copy

class CyclicWalk(Dataset):

    def __init__(self, path, time_bin=1000000, velocity_threshold=0.1):

        super().__init__()
        self.name = "cyclic-walk"
        self.velocity_threshold = velocity_threshold
        
        mat = utils.loadmat(path)
        rosdata = mat["x"]["rosdata"]
        enctimes = rosdata["encTimes"]
        velocity = mat["x"]["rosdata"]["vel"]
        encangle = rosdata["encAngle"]
        n_cells = len(mat["x"]["clust"])
                
        times = self.get_times(mat)
        
        # Bin times
        regular_times = np.arange(start=times[0], stop=times[-1], step=time_bin)
        n_times = len(regular_times) - 1
        place_cells = np.zeros((n_times, n_cells))

        for i_cell, cell in tqdm(enumerate(mat["x"]["clust"])):
            # print(f"Processing cell {i_cell}...")
            counts, bins, _ = plt.hist(cell["ts"], bins=regular_times)
            assert sum(bins != regular_times) == 0
            assert len(counts) == n_times
            place_cells[:, i_cell] = counts
            
        # Standardize the Data
        place_cells = place_cells - place_cells.mean(axis=-1, keepdims=True)
        place_cells = place_cells / (np.linalg.norm(place_cells, axis=-1, keepdims=True) + 1e-10)
                
        enc_counts, enc_bins = np.histogram(enctimes, bins=regular_times)
        
        # Bin Position Angles
        angles = []
        cum_count = 0
        for count in enc_counts:
            angles.append(np.mean(encangle[cum_count:cum_count+int(count)]))
            cum_count += int(count)
        assert len(angles) == len(regular_times) -1
        angles = [x % 360 for x in angles]
        angles = torch.tensor([np.deg2rad(x) for x in angles])
        
        # Bin Velocity
        velocities = []
        cum_count = 0
        for count in enc_counts:
            velocities.append(np.mean(velocity[cum_count:cum_count+int(count)]))
            cum_count += int(count)
        assert len(velocities) == len(regular_times) -1
        velocities = torch.tensor(velocities)
        
        vel_idx = abs(velocities) >= velocity_threshold
        place_cells = place_cells[vel_idx]
        angles = angles[vel_idx]
        
        good_idx = np.where(place_cells.max(axis=-1) != 0.0)
        place_cells = place_cells[good_idx]
        angles = angles[good_idx]
        velocities = velocities[good_idx]
        
        self.velocity = velocities
        self.dim = place_cells.shape[1]
        self.data = torch.tensor(place_cells, dtype=torch.float32)
        self.labels = torch.tensor(angles, dtype=torch.float32)
        
    def get_times(self, mat):
        times = []
        for clust in mat["x"]["clust"]:
            times.extend(clust["ts"])

        times = sorted(times)
        n_times = len(times)
        # print(f"Number of times before deleting duplicates: {n_times}.")
        aux = []
        for time in times:
            if time not in aux:
                aux.append(time)
        n_times = len(aux)
        # print(f"Number of times after deleting duplicates: {n_times}.")
        times = aux
        times = np.array(times)
        return times

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]
        return x, y

    def __len__(self):
        return len(self.data)
    
    

class CyclicWalkAngle(Dataset):

    def __init__(self, path, randomize_pairs=True, time_bin=1000000, velocity_threshold=0.1):

        super().__init__()
        self.name = "cyclic-walk"
        self.randomize_pairs = randomize_pairs
        self.velocity_threshold = velocity_threshold
        
        mat = utils.loadmat(path)
        rosdata = mat["x"]["rosdata"]
        enctimes = rosdata["encTimes"]
        velocity = mat["x"]["rosdata"]["vel"]
        encangle = rosdata["encAngle"]
        n_cells = len(mat["x"]["clust"])
                
        times = self.get_times(mat)
        
        # Bin times
        regular_times = np.arange(start=times[0], stop=times[-1], step=time_bin)
        n_times = len(regular_times) - 1
        place_cells = np.zeros((n_times, n_cells))

        for i_cell, cell in tqdm(enumerate(mat["x"]["clust"])):
            # print(f"Processing cell {i_cell}...")
            counts, bins, _ = plt.hist(cell["ts"], bins=regular_times)
            assert sum(bins != regular_times) == 0
            assert len(counts) == n_times
            place_cells[:, i_cell] = counts
            
        # Standardize the Data
        place_cells = place_cells - place_cells.mean(axis=-1, keepdims=True)
        place_cells = place_cells / (np.linalg.norm(place_cells, axis=-1, keepdims=True) + 1e-10)
                
        enc_counts, enc_bins = np.histogram(enctimes, bins=regular_times)
        
        # Bin Position Angles
        angles = []
        cum_count = 0
        for count in enc_counts:
            angles.append(np.mean(encangle[cum_count:cum_count+int(count)]))
            cum_count += int(count)
        assert len(angles) == len(regular_times) -1
        angles = [x % 360 for x in angles]
        angles = torch.tensor([np.deg2rad(x) for x in angles])
        
        # Bin Velocity
        velocities = []
        cum_count = 0
        for count in enc_counts:
            velocities.append(np.mean(velocity[cum_count:cum_count+int(count)]))
            cum_count += int(count)
        assert len(velocities) == len(regular_times) -1
        velocities = torch.tensor(velocities)
        
        vel_idx = abs(velocities) >= velocity_threshold
        place_cells = place_cells[vel_idx]
        angles = angles[vel_idx]
        
        good_idx = np.where(place_cells.max(axis=-1) != 0.0)
        place_cells = place_cells[good_idx]
        angles = angles[good_idx]
        velocities = velocities[good_idx]
        
        if self.randomize_pairs:
            data, data_next, angle_diff = self.construct_dataset_r(place_cells, angles)

        else:
            data, data_next, angle_diff = self.construct_dataset(place_cells, angles)
            
        self.velocity = velocities
        self.dim = data.shape[1]
        self.data = torch.tensor(data, dtype=torch.float32)
        self.data_next = torch.tensor(data_next, dtype=torch.float32)
        self.angle = torch.tensor(angle_diff, dtype=torch.float32)
        self.pos = angles
        
    def get_times(self, mat):
        times = []
        for clust in mat["x"]["clust"]:
            times.extend(clust["ts"])

        times = sorted(times)
        n_times = len(times)
        # print(f"Number of times before deleting duplicates: {n_times}.")
        aux = []
        for time in times:
            if time not in aux:
                aux.append(time)
        n_times = len(aux)
        # print(f"Number of times after deleting duplicates: {n_times}.")
        times = aux
        times = np.array(times)
        return times
    
    def construct_dataset(self, time_series, angles):
        data = time_series[:-1]
        angle_t = angles[:-1]
        data_t1 = time_series[1:]
        angle_t1 = angles[1:]
        angle_diff = []
        for i, dt in enumerate(data):
            diff = (angle_t1[i] - angle_t[i]) % (2 * np.pi)
            angle_diff.append(diff)
        return data, data_t1, angle_diff
    
    def construct_dataset_r(self, time_series, angles):
        data = time_series
        angle_t = angles
        idx = np.arange(len(data))
        np.random.shuffle(idx)
        # idx = torch.tensor(idx)
        data_t1 = time_series[idx]
        angle_t1 = angles[idx]
        angle_diff = []
        for i, dt in enumerate(data):
            diff = (angle_t1[i] - angle_t[i]) % (2 * np.pi)
            angle_diff.append(diff)
        return data, data_t1, angle_diff

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.data_next[idx]
        angle = self.angle[idx]
        return x, y, angle

    def __len__(self):
        return len(self.data)
    
    
class CyclicWalkAngleLoader:
    def __init__(self, 
                 batch_size, 
                 fraction_val=0.2,
                 num_workers=0, 
                 seed=0):
        assert (
            fraction_val <= 1.0 and fraction_val >= 0.0
        ), "fraction_val must be a fraction between 0 and 1"

        np.random.seed(seed)

        self.batch_size = batch_size
        self.fraction_val = fraction_val
        self.seed = seed
        self.num_workers = num_workers
        
    def split_data(self, dataset):
        
        if self.fraction_val > 0.0:
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            split = int(np.floor(self.fraction_val * len(dataset)))

            np.random.shuffle(indices)

            train_indices, val_indices = indices[split:], indices[:split]
            val_dataset = copy.deepcopy(dataset)
            val_dataset.data = val_dataset.data[val_indices]
            val_dataset.data_next = val_dataset.data_next[val_indices]
            val_dataset.angle = val_dataset.angle[val_indices]
            
            train_dataset = copy.deepcopy(dataset)
            train_dataset.data = train_dataset.data[train_indices]
            train_dataset.data_next = train_dataset.data_next[train_indices]
            train_dataset.angle = train_dataset.angle[train_indices]
        
        else:
            val_dataset = None
    
        return train_dataset, val_dataset
    
    def construct_data_loaders(self, train_dataset, val_dataset):
        if val_dataset is not None:
            val = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
        
        else:
            val = None
            
        train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train, val         

    def load(self, dataset):
        train_dataset, val_dataset = self.split_data(dataset)
        self.train, self.val = self.construct_data_loaders(train_dataset, val_dataset)