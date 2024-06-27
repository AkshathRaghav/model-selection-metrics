# --- 
# Adapted from: https://github.com/TencentARC/SFDA/
# ---

from ptmrank.metrics.Metric import Metric, MetricError
from ptmrank.tools.logger import LoggerSetup
import numpy as np
import torch
from numba import njit
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def _NLEEP(X, y, component_ratio=5):

    n = len(y)
    num_classes = len(np.unique(y))
    # PCA: keep 80% energy
    pca_80 = PCA(n_components=0.8)
    pca_80.fit(X)
    X_pca_80 = pca_80.transform(X)

    # GMM: n_components = component_ratio * class number
    n_components_num = component_ratio * num_classes
    gmm = GaussianMixture(n_components= n_components_num).fit(X_pca_80)
    prob = gmm.predict_proba(X_pca_80)  # p(z|x)
    
    # NLEEP
    pyz = np.zeros((num_classes, n_components_num))
    for y_ in np.unique(y):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n   

    pz = np.sum(pyz, axis=0)    
    py_z = pyz / pz             
    py_x = np.dot(prob, py_z.T) 

    # nleep_score
    nleep_score = np.sum(py_x[np.arange(n), y]) / n
    return nleep_score

class NLEEP(Metric):
    """
    NLEEP calculation as proposed in the CVPR 2018 paper
    "Ranking Neural Checkpoints"
    from https://arxiv.org/pdf/2011.11200v4 
    """
    def __init__(self):
        self.logger = LoggerSetup("Metric [NLEEP]").get_logger()
        self.logger.info("Booted: Metric [NLEEP].")

    def __str__(self):
        return "NLEEP"

    def reset(self):    
        self.embeddings = None
        self.targets = None
        self.class_labels = None
    
    def test(self): 
        self.logger.info("Running test.")

        dim = 1024
        embeddings = torch.rand(1000, dim)
        targets = torch.randint(0, 3, (1000,))

        score2 = _NLEEP(embeddings, targets)
        self.initialize(embeddings, targets)
        score1 = self.fit()
        self.reset() 

        if not np.isclose(score1, score2, atol=1e-2):
            raise MetricError(f"Test failed. NLEEP: {score1:2f} vs. Real NLEEP(): {score2:.2f}")
        
        self.logger.info("Success.")

    def initialize(self, embeddings, targets) -> None:

        super().__init__("NLEEP", embeddings, targets)

        self.logger.info("Applying PCA to Embeddings.")
        self.embeddings = self.apply_PCA(embeddings, 0.8)
        self.logger.info("PCA Applied.")

        self.logger.info("Initialization Complete.")
    
    def fit(self): 
        """ 
        self.targets: shape [N, C_s]; from source
        self.embeddings: shape [N, D]; from target
        """
        y = self.targets
        n = len(self.targets)
        num_classes = len(self.class_labels)

        n_components_num = 5 * num_classes
        gmm = GaussianMixture(n_components=n_components_num).fit(self.embeddings)
        prob = torch.from_numpy(gmm.predict_proba(self.embeddings)).float()  # p(z|x)

        pyz = torch.zeros(num_classes, n_components_num)
        for y_ in self.class_labels:
            a = torch.sum(prob[y==y_], dim=(0,))
            pyz[y_] = a / n  
        py_z = pyz / torch.sum(pyz, dim=(0,))
        py_x = torch.matmul(prob, py_z.T) 

        nleep_score = torch.sum(py_x[np.arange(n), y]) / n
        self.logger.info(f"NLEEP: {nleep_score:.2f}")
        return nleep_score

NLEEP().test()