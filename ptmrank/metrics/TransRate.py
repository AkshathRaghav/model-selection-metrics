# Adapted from: https://github.com/Long-Kai/TransRate/blob/master/model_selection.py


import numpy as np 
import torch 
from .Metric import Metric
from ..tools.logger import LoggerSetup

class TransRate(Metric):
    """
    TransRate Score calculation as proposed in 
    "Frustratingly easy transferability estimation"
    from https://arxiv.org/abs/2106.09362
    """
    def __init__(self):
        self.logger = LoggerSetup("Metric [TransRate]").get_logger()
        self.logger.info("Booted: Metric [TransRate].")

    def __str__(self):
        return "TransRate"

    def reset(self):
        self.embeddings = None
        self.targets = None
        self.class_labels = None
        self.class_label_counts = None 
    
    def test(self): 
        self.logger.info("Running test.")

        dim = 1024 
        embeddings = torch.rand(1000, dim)
        targets = torch.randint(0, 3, (1000,))

        self.initialize(embeddings, targets)
        _ = self.fit() 
        self.reset() 

        self.logger.info("Success.")

    def initialize(self, embeddings, targets, eps=1e-4, reg=1., opt_active_eigs='all', opt_res='weight', normalize_eigz=False, normalize_opt='rank'): 
        self.logger.info("Initializing TransRate.")
        super().__init__(embeddings, targets)

        self.eps = eps
        self.reg = reg
        self.opt_active_eigs = opt_active_eigs
        self.opt_res = opt_res
        self.normalize_eigz = normalize_eigz
        self.normalize_opt = normalize_opt

        for i in range(len(self.class_labels)):
            self.logger.info(f"Class {self.class_labels[i]} has {self.class_label_counts[i]} samples.")

        self.logger.info("Initialization Complete.")

    def fit(self): 
        z = self.embeddings 
        y = self.targets

        if self.use_proj_Z:
            eig_Z, rank_Z, eig_Zc, rank_Zc, n_Zc = pre_transrate(z_centralized, y)
        else:
            z_centralized = z - np.mean(z, axis=0, keepdims=True).repeat(len(z), axis=0)
            eig_Z, rank_Z, eig_Zc, rank_Zc, n_Zc = pre_transrate_low_dim_proj(z, y, opt='mul', opt_new=False)

        return _transrate(eig_Z, rank_Z, eig_Zc, rank_Zc, n_Zc)

def _transrate(self, eig_Z, rank_Z, eig_Zc, rank_Zc, n_Zc): 
    n = np.sum(n_Zc)
    if self.normalize_eigz:
        sum_eig_z = np.sum(eig_z)
        eig_z = eig_z / sum_eig_z
        eig_zc = eig_zc / sum_eig_z

    d = eig_z.shape[0]

    if self.opt_active_eigs == 'rank':
        idx = self.rank_z
    elif self.opt_active_eigs == 'all':
        idx = d

    if self.opt_res == 'avg':
        eig_z[:idx] = eig_z[:idx] + np.sum(eig_z[idx:]) / idx
    elif opt_res == 'weight':
        eig_z[:idx] = eig_z[:idx] + np.sum(eig_z[idx:]) * eig_z[:idx] / np.sum(eig_z[:idx])
    else:
        raise ValueError("Not supported opt res")

    rz = np.sum(np.log(self.reg + 1/self.eps * np.abs(eig_z[:idx])))


    nClass = len(self.rank_zc)

    rzc = np.zeros(nClass)

    for i in range(nClass):
        eig_zc_i = eig_zc[i]

        if self.opt_res == 'avg':
            eig_zc_i[:idx] = eig_zc_i[:idx] + np.sum(eig_zc_i[idx:]) / idx
        elif self.opt_res == 'weight':
            eig_zc_i[:idx] = eig_zc_i[:idx] + np.sum(eig_zc_i[idx:]) * eig_zc_i[:idx] / np.sum(eig_zc_i[:idx])

        rzc[i] = np.sum(np.log(self.reg + 1/self.eps * np.abs(eig_zc_i[:idx])))

    if self.normalize_opt == 'dim':
        normal_r = min(d, idx)
        rz = rz / normal_r
        rzc = rzc / normal_r
    elif self.normalize_opt == 'rank':
        normal_r = min(self.rank_z, idx)
        rz = rz / normal_r
        rzc = rzc / normal_r

    return rz/2. - np.sum(rzc * np.array(n_Zc)/n)/2.

def pre_transrate_low_dim_proj(f, y):
    n, d = f.shape
    Z = f

    mean_f = np.mean(f, axis=0)
    mean_f = np.expand_dims(mean_f, 1)

    covf = f.T @ f / n - mean_f * mean_f.T

    K = int(y.max() + 1)

    eig_c = []
    nc = []
    rank_c = []
    cov_Zi = []

    for i in range(K):
        y_ = (y == i).flatten()
        Zi = Z[y_]
        nci = Zi.shape[0]
        nc.append(nci)

        mean_Zi = np.mean(Zi, axis=0)
        mean_Zi = np.expand_dims(mean_Zi, 1)
        g = mean_Zi - mean_f

        if i == 0:
            covg = g @ g.T * nci
        else:
            covg = g @ g.T * nci + covg

        ZZi = Zi.T @ Zi - (mean_Zi * mean_Zi.T) * nci

        cov_Zi.append(ZZi)

    proj_mat = np.dot(np.linalg.pinv(covf, rcond=1e-15), covg/n)


    for i in range(K):
        eigs_i, ri = eigens_and_rank(cov_Zi[i] @ proj_mat / nc[i])
        eig_c.extend(np.expand_dims(eigs_i, axis=0))
        rank_c.append(ri)
    eig_Z, rank_Z = eigens_and_rank(covf @ proj_mat)

    return eig_Z, rank_Z, np.stack(eig_c), rank_c, nc

def pre_transrate(f, y, normalize=True):
    if normalize:
        l2 = np.atleast_1d(np.linalg.norm(f, 2, -1))
        Z = f / np.expand_dims(l2, -1)
    else:
        Z = f

    K = int(y.max() + 1)

    n, d = f.shape

    eig_c = []
    nc = []
    rank_c = []

    for i in range(K):
        y_ = (y == i).flatten()
        Zi = Z[y_]

        ZZi = Zi.transpose() @ Zi
        if i == 0:
            ZZ = ZZi
        else:
            ZZ = ZZ + ZZi

        nc.append(Zi.shape[0])

        eigs_i, ri = eigens_and_rank(ZZi / float(Zi.shape[0]))

        eig_c.extend(np.expand_dims(eigs_i, axis=0))
        rank_c.append(ri)

    ZZ = ZZ / float(n)
    eig_Z, rank_Z = eigens_and_rank(ZZ)

    return eig_Z, rank_Z, np.stack(eig_c), rank_c, nc

def eigens_and_rank(ZZ):
    _, eigs, _ = np.linalg.svd(ZZ, full_matrices=False)
    r = np.linalg.matrix_rank(ZZ)

    return eigs, r