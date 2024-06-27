# --- 
# Adapted from: https://github.com/YaojieBao/An-Information-theoretic-Metric-of-Transferability
# ---


import numpy as np 
from ptmrank.metrics.Metric import Metric, MetricError
from ptmrank.tools.logger import LoggerSetup
from numba import njit
import torch 
from ..tools.cov import cov 
from sklearn.covariance import LedoitWolf



def h_score(features: np.ndarray, labels: np.ndarray):
    r"""
    H-score in `An Information-theoretic Approach to Transferability in Task Transfer Learning (ICIP 2019) 
    <http://yangli-feasibility.com/home/media/icip-19.pdf>`_.
    
    The H-Score :math:`\mathcal{H}` can be described as:

    .. math::
        \mathcal{H}=\operatorname{tr}\left(\operatorname{cov}(f)^{-1} \operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)
    
    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector

    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.

    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    f = features
    y = labels

    covf = np.cov(f, rowvar=False)
    C = int(y.max() + 1)
    g = np.zeros_like(f)

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)
    score = np.trace(np.dot(np.linalg.pinv(covf, rcond=1e-15), covg))

    return score

def regularized_h_score(features: np.ndarray, labels: np.ndarray):
    r"""
    Regularized H-score in `Newer is not always better: Rethinking transferability metrics, their peculiarities, stability and performance (NeurIPS 2021) 
    <https://openreview.net/pdf?id=iz_Wwmfquno>`_.
    
    The  regularized H-Score :math:`\mathcal{H}_{\alpha}` can be described as:

    .. math::
        \mathcal{H}_{\alpha}=\operatorname{tr}\left(\operatorname{cov}_{\alpha}(f)^{-1}\left(1-\alpha \right)\operatorname{cov}\left(\mathbb{E}[f \mid y]\right)\right)
    
    where :math:`f` is the features extracted by the model to be ranked, :math:`y` is the groud-truth label vector and :math:`\operatorname{cov}_{\alpha}` the  Ledoit-Wolf 
    covariance estimator with shrinkage parameter :math:`\alpha`
    Args:
        features (np.ndarray):features extracted by pre-trained model.
        labels (np.ndarray):  groud-truth labels.

    Shape:
        - features: (N, F), with number of samples N and feature dimension F.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar.
    """
    f = features.astype('float64')
    f = f - np.mean(f, axis=0, keepdims=True)  # Center the features for correct Ledoit-Wolf Estimation
    y = labels

    C = int(y.max() + 1)
    g = np.zeros_like(f)

    cov = LedoitWolf(assume_centered=False).fit(f)
    alpha = cov.shrinkage_
    covf_alpha = cov.covariance_

    for i in range(C):
        Ef_i = np.mean(f[y == i, :], axis=0)
        g[y == i] = Ef_i

    covg = np.cov(g, rowvar=False)
    score = np.trace(np.dot(np.linalg.pinv(covf_alpha, rcond=1e-15), (1 - alpha) * covg))

    return score

class HScore(Metric): 
    """ 
    H-Score calculation as proposed in the ICIP 2019 paper
    "An Information-theoretic Metric of Transferability" by Yaojie Bao, et al.
    from https://ieeexplore.ieee.org/document/8803726

    Note: 
        - Higher H-Score indicates better transferability. 
    """

    def __init__(self): 
        self.logger = LoggerSetup("Metric [H-Score]").get_logger()
        self.logger.info("Initializing Metric [H-Score].")

    def __str__(self):
        return "H-Score"
    
    def reset(self):
        self.embeddings = None
        self.targets = None
    
    def test(self): 
        self.logger.info("Running test.")

        dim = 1024
        embeddings = torch.rand(1000, dim)
        targets = torch.randint(0, 3, (1000,))

        self.initialize(embeddings, targets)
        score1 = self.fit()
        score2 = h_score(embeddings.detach().numpy(), targets.detach().numpy())
        self.reset()
        
        if not np.isclose(score1, score2, atol=1e-2):
            raise MetricError(f"Test failed. H-Score: {score1:.2f} vs. h_score(): {score2:.2f}")
        
        self.logger.info("Success.")

    def initialize(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        super().__init__("H-Score", embeddings, targets)

        self.logger.info("Initializing H-Score.")

        unique, counts = np.unique(targets, return_counts=True)
        for i in range(len(unique)):
            self.logger.info(f"Class {unique[i]} has {counts[i]} samples.")
        self.classes = unique

        self.logger.info(f"Shape of final featurs: {self.embeddings.shape}")
        self.logger.info("Initialization Complete.")

    @staticmethod    
    def getHscore(f,Z,z_):
        Covf=cov(f, rowvar=False)
        
        g=torch.zeros_like(f)
        for z in z_:
            g[Z == z]=torch.mean(f[Z == z], dim=(0,))
        
        Covg=cov(g, rowvar=False)
        score=torch.trace(torch.matmul(torch.linalg.pinv(Covf,rcond=1e-15), Covg))
        return score

    def fit(self) -> float:
        self.logger.info("Calculating H-Score.")
        score = HScore.getHscore(self.embeddings, self.targets, self.classes)
        self.logger.info(f"H-Score: {score:.2f}")
        return score

class HAlpha_Score(Metric): 
    """ 
    Regularized H-score calculation as proposed in the NeurIPS 2021 paper
    "Newer is not always better: Rethinking transferability metrics" 
    from https://openreview.net/pdf?id=iz_Wwmfquno
    """

    def __init__(self): 
        self.logger = LoggerSetup("Metric [H_Alpha-Score]").get_logger()
        self.logger.info("Initializing Metric [H_Alpha-Score].")

    def __str__(self):
        return "H_Alpha-Score"
    
    def reset(self):
        self.embeddings = None
        self.targets = None
    
    def test(self): 
        """ 
        Tests to confirm compilation and lack of runtime errors. Does not validate the correctness of the metric.
        """
        
        self.logger.info("Running test.")

        dim = 1024
        embeddings = torch.rand(1000, dim)
        targets = torch.randint(0, 3, (1000,))

        self.initialize(embeddings, targets)
        score1 = self.fit()
        score2 = regularized_h_score(embeddings.detach().numpy(), targets.detach().numpy())
        self.reset()
        
        if not np.isclose(score1, score2, atol=1e-2):
            raise MetricError(f"Test failed. Regularized H-Score: {score1:.2f} vs. regularized_h_score(): {score2:.2f}")
        
        self.logger.info("Success.")

    def initialize(self, embeddings: np.ndarray, targets: np.ndarray) -> None:
        super().__init__("H_Alpha-Score", embeddings, targets)

        self.logger.info("Initializing H_Alpha-Score.")

        unique, counts = np.unique(targets, return_counts=True)
        for i in range(len(unique)):
            self.logger.info(f"Class {unique[i]} has {counts[i]} samples.")
        self.classes = unique

        self.logger.info(f"Shape of final featurs: {self.embeddings.shape}")
        self.logger.info("Initialization Complete.")

    @staticmethod
    def getHAlphascore(shrinkage,covariance,f,Z,z_):
        g = torch.zeros_like(f)
        for z in z_:
            g[Z == z]=torch.mean(f[Z == z], dim=(0,))
        
        Covg=cov(g, rowvar=False)
        score=torch.trace(torch.matmul(torch.linalg.pinv(covariance, rcond=1e-15), (1 - shrinkage) * Covg))
        return score

    def fit(self) -> float:
        self.logger.info("Calculating H-Score.")
        centered_embeddings = self.embeddings - torch.mean(self.embeddings, dim=0, keepdims=True)
        shrunk_cov = LedoitWolf(assume_centered=False).fit(centered_embeddings)
        score = HAlpha_Score.getHAlphascore(shrunk_cov.shrinkage_, torch.from_numpy(shrunk_cov.covariance_), self.embeddings, self.targets, self.classes)
        
        self.logger.info(f"Regularized H-Score: {score:.2f}")
        return score


# HScore().test()
# HAlpha_Score().test()