import torch
import math
import numpy as np
import pandas as pd
import os

from torchmetrics.functional import spearman_corrcoef, pearson_corrcoef
from torchmetrics.functional import concordance_corrcoef, explained_variance

from .structural import convert_to_tensor

from netrep.metrics import LinearMetric
import itertools
import multiprocessing
from tqdm import tqdm

### RSA: Compare RDMS (code by C. Conwell) ------------------------------------------------------------

_compare_rdms_by = {'spearman': spearman_corrcoef,
                    'pearson': pearson_corrcoef,
                    'concordance': concordance_corrcoef}

def extract_rdv(X):
    return X[torch.triu(torch.ones_like(X, dtype=bool), diagonal=1)]

def compare_rdms(rdm1, rdm2, method = 'pearson', device = None, **method_kwargs):
    rdm1, rdm2 = convert_to_tensor(rdm1, rdm2)
    
    if device is None:
        rdm2 = rdm2.to(rdm1.device)
    else:
        rdm1 = rdm1.to(device)
        rdm2 = rdm2.to(device)
        
    rdm1_triu = extract_rdv(rdm1)
    rdm2_triu = extract_rdv(rdm2)
    
    return _compare_rdms_by[method](rdm1_triu, rdm2_triu, **method_kwargs).item()

def fisherz(r, eps=1e-5):
    return torch.arctanh(r-eps)

def fisherz_inv(z):
    return torch.tanh(z)

def average_rdms(rdms):
    return (1 - fisherz_inv(fisherz(torch.stack([1 - rdm for rdm in rdms]))
                            .mean(axis = 0, keepdims = True).squeeze()))



### RSA: Calculate RDMS (code by C. Conwell) -----------------------------------------------------------

def compute_rdm(data, method = 'pearson', norm=False, device=None, **rdm_kwargs):
    rdm_args = (data, norm, device)
    if method == 'euclidean':
        return compute_euclidean_rdm(*rdm_args, **rdm_kwargs)
    if method == 'pearson':
        return compute_pearson_rdm(*rdm_args, **rdm_kwargs)
    if method == 'spearman':
        return compute_spearman_rdm(*rdm_args, **rdm_kwargs)
    if method == 'mahalanobis':
        return compute_mahalanobis_rdm(*rdm_args, **rdm_kwargs)
    if method == 'concordance':
        return compute_concordance_rdm(*rdm_args, **rdm_kwargs)

def compute_mahalanobis_rdm(data, norm = False, device=None):
    data = convert_to_tensor(data, device=device)
    cov_matrix = torch.cov(data.T)
    inv_cov_matrix = torch.inverse(cov_matrix)
    centered_data = data - torch.mean(data, axis=0)
    kernel = centered_data @ inv_cov_matrix @ centered_data.T
    rdm = torch.diag(kernel).unsqueeze(1) + torch.diag(kernel).unsqueeze(0) - 2 * kernel
    return rdm / data.shape[1] if norm else rdm
    
def compute_pearson_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    rdm = 1 - torch.corrcoef(data)
    return rdm / data.shape[1] if norm else rdm

def compute_spearman_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    rank_data = data.argsort(dim=1).argsort(dim=1)
    rdm = 1 - torch.corrcoef(rank_data)
    return rdm / data.shape[1] if norm else rdm

def compute_euclidean_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    rdm = torch.cdist(data, data, p=2.0)
    rdm = rdm.fill_diagonal_(0)
    return rdm / data.shape[1] if norm else rdm

def compute_concordance_rdm(data, norm=False, device=None):
    data = convert_to_tensor(data, device=device)
    mean_matrix = data.mean(dim=1)
    var_matrix = data.var(dim=1)
    std_matrix = data.std(dim=1)
    corr_matrix = torch.corrcoef(data)
    numerator = 2 * corr_matrix * std_matrix[:, None] * std_matrix[None, :]
    denominator1 = var_matrix[:, None] + var_matrix[None, :]
    denominator2 = (mean_matrix[:, None] - mean_matrix[None, :]) ** 2
    rdm =  1 - (numerator / (denominator1 + denominator2))
    return rdm / data.shape[1] if norm else rdm

def get_rdms_by_indices(data, indices, method='pearson', device='cpu', **rdm_kwargs):
    def get_rdm(data, index):
        if isinstance(data, (np.ndarray, torch.Tensor)):
            rdm_data = data[index, :].T
        if isinstance(data, pd.DataFrame):
            rdm_data = data.loc[index].to_numpy().T
        return compute_rdm(rdm_data, method, device=device, **rdm_kwargs)

    if isinstance(indices, (np.ndarray, torch.Tensor)):
        return get_rdm(data, indices)
    if isinstance(indices, dict):
        rdms_dict = {}
        for key, index in indices.items():
            rdms_dict[key] = get_rdms_by_indices(data, index, method, device, **rdm_kwargs)
        return rdms_dict
    
def clean_nan_rdms(rdm_data):
    def rdm_nan_check(rdm):
        return rdm if torch.sum(torch.isnan(rdm) == 0) else None
    
    if isinstance(rdm_data, (np.ndarray, torch.Tensor)):
        return rdm_nan_check(rdm_data)

    if isinstance(rdm_data, dict):
        cleaned_dict = {}
        for key, data in rdm_data.items():
            cleaned_data = clean_nan_rdms(data)
            if cleaned_data is not None:  
                cleaned_dict[key] = cleaned_data
        return cleaned_dict
    
def get_traintest_rdms(rdm_data, test_idx=None):
    def get_traintest_rdm(rdm, test_idx):
        if test_idx is not None:
            if isinstance(test_idx, np.ndarray):
                test_idx = torch.tensor(test_idx)
                
            train_idx = torch.ones(rdm.shape[0], dtype=torch.bool)
            train_idx[test_idx] = False

            return {'train': rdm[train_idx, train_idx], 
                    'test': rdm[test_idx, test_idx]}
        
        return {'train': rdm[::2, ::2], 'test': rdm[1::2, 1::2]}

    if isinstance(rdm_data, (np.ndarray, torch.Tensor)):
        return get_traintest_rdm(rdm_data, test_idx)

    if isinstance(rdm_data, dict):
        rdms_dict = {}
        for key, data in rdm_data.items():
            rdms_dict[key] = get_traintest_rdms(data, test_idx)
        return rdms_dict
    



### (code by F. Acosta)

def _compute_rsa_dissimilarity(i, j, rdm1, rdm2):
    rdm1.cpu()
    rdm2.cpu()
    rsa_dissimilarity = 1 - compare_rdms(rdm1, rdm2, method="pearson")
    return i, j, rsa_dissimilarity


def _compute_rsa_dissimilarity_star(args):
    return _compute_rsa_dissimilarity(*args)

def compute_rsa_pairwise_dissimilarities(neural_data, processes=None):
    functional_rois = list(neural_data.keys())
    rdms = {}
    for region in functional_rois:
        rdms[region] = compute_rdm(
            neural_data[region].to_numpy().transpose(), method="pearson"
        )
    rdms_list = list(rdms.values())
    
    n = len(functional_rois)

    n_dists = n*(n-1)/2

    ij = itertools.combinations(range(n),2)
    args = ((i,j, rdms_list[i], rdms_list[j]) for i, j in ij)

    print(f"Parallelizing n(n-1)/2 = {int(n_dists)} distance calculations with {multiprocessing.cpu_count() if processes is None else processes} processes.")
    pbar = lambda x: tqdm(x, total=n_dists, desc="Computing distances")

    with multiprocessing.pool.ThreadPool(processes=processes) as pool:
        results = []
        for result in pbar(pool.imap_unordered(_compute_rsa_dissimilarity_star, args)):
            results.append(result)



    rsa_pairwise_dissimilarity = np.zeros((n,n))

    for i, j, rsa_dissimilarity in results:
        rsa_pairwise_dissimilarity[i,j], rsa_pairwise_dissimilarity[j,i] = rsa_dissimilarity, rsa_dissimilarity

    return rsa_pairwise_dissimilarity


### CKA Methods (code by C. Conwell) -----------------------------------------------------------

class TorchCKA():
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)
    
    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)
    
    @staticmethod
    def cite_source():
        print('See: https://github.com/jayroxis/CKA-similarity')




### Shape-Based Metrics (code by F. Acosta, adapted from https://github.com/ahwillia/netrep) -----------------------------------------------------------

def train_test_split(neural_data, stimulus, seed=2023):
    """Create 80/20 train/test data split.

    Parameters
    ----------
    neural_data : array-like, shape = []
        fMRI voxel activations.
    stimulus : 
        stimulus data (NSD images)
    
    Returns
    -------
    train_data :  
    test_data : 
    """

    n_features, n_classes = next(iter(neural_data.values())).shape
    functional_rois = list(neural_data.keys())

    seed = 2023
    rng = np.random.default_rng(seed)

    idx_train = rng.choice(np.arange(n_classes), int(n_classes * 0.8), replace=False)
    train_images = stimulus.loc[idx_train]["image_id"].astype(str)

    idx_test = np.array(list(set(np.arange(n_classes)).difference(idx_train)))
    test_images = stimulus.loc[idx_test]["image_id"].astype(str)

    train_dict = {
        region: neural_data[region][train_images].to_numpy().T
        for region in functional_rois
    }
    
    test_dict = {
        region: neural_data[region][test_images].to_numpy().T
        for region in functional_rois
    }

    train_data = list(train_dict.values())
    test_data = list(test_dict.values())

    return train_data, test_data


def compute_pairwise_distances(neural_data, stimulus, alpha = 1):
    """Compute matrix of pairwise shape-space distances between N networks.

    Parameters
    ----------
    neural_data : array-like, shape=[]
        fMRI voxel activations.
    alpha : float, 0 < alpha < 1
        Paramaterizes metric. alpha=0 -> affine invariance, alpha=1 -> Procrustes distance.

    Returns
    -------
    pairwise_distance_matrix : array-like, shape=[N,N]
        NxN matrix of pairwise distances between N network representations. 
    """
    
    os.environ["OMP_NUM_THREADS"] = "1"

    metric = LinearMetric(alpha=alpha, center_columns=True, score_method="angular")

    train_data, test_data = train_test_split(neural_data, stimulus)

    n = len(train_data)
    print(f"We have n = {n} cortical regions;")
    print(f"We need n(n-1)/2 = {int((n*(n-1)/2))} distance calculations")

    pairwise_dist_train, pairwise_dist_test = metric.pairwise_distances(train_data, test_data)

    return pairwise_dist_test




