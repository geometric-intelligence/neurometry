### NSD data -------
target_regions = ["OTC"]  # , "EVC"]
subjects = ["subj01", "subj02", "subj05", "subj07"]
#subjects = ["subj01"]


### RSA settings ----------
rdm_compute_types = ["euclidean", "pearson", "spearman", "mahalanobis", "concordance"]
rdm_compare_types = ["pearson", "spearman", "concordance"]

# rdm_compute_types = ["pearson"]
# rdm_compare_types = ["pearson"]

### Shape metrics settings ----------
alphas = [0, 0.5, 1]

### CKA settings ----------
cka_types = ["linear_cka", "kernel_cka"]


### boostrapping
n_bootstrap_iterations = 4
