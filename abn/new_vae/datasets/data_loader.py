import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import numpy as np
import copy


class TrainValLoader:
    def __init__(self, batch_size, fraction_val=0.2, num_workers=0, seed=0):
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
            val_dataset.labels = val_dataset.labels[val_indices]

            train_dataset = copy.deepcopy(dataset)
            train_dataset.data = train_dataset.data[train_indices]
            train_dataset.labels = train_dataset.labels[train_indices]

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
                pin_memory=True,
            )

        else:
            val = None

        train = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        return train, val

    def load(self, dataset):
        train_dataset, val_dataset = self.split_data(dataset)
        self.train, self.val = self.construct_data_loaders(train_dataset, val_dataset)
