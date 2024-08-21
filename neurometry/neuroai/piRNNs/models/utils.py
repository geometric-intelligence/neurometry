import numpy as np
import torch


def average_appended_metrics(metrics):
    ks = metrics[0].keys()
    return {k: np.mean([metrics[i][k] for i in range(len(metrics))]) for k in ks}


def dict_to_numpy(data):
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = dict_to_numpy(data[key])
        else:
            data[key] = data[key].cpu().detach().numpy()

    return data


def dict_to_device(data, device):
    for key, value in data.items():
        if isinstance(value, dict):
            data[key] = dict_to_device(data[key], device)
        else:
            data[key] = torch.tensor(data[key]).to(device).float()

    return data
