# Adapted from: https://github.com/TencentARC/SFDA
# Note: Most of the code depends upon tools::lda::LDA, with some changes. 

import numpy as np
from scipy import linalg
from utils import iterative_A

import torch 
from .Metric import Metric
from ..tools.helpers import iterative_A
from ..tools.logger import LoggerSetup
from ..tools.lda import LDA 

class _SFDA(LDA):
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        super().__init__(shrinkage, priors, n_components)
        
    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        return super().fit(X, y)
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

class SFDA(Metric): 
    """
    SFDA Score calculation as proposed in 
    "Not All Models Are Equal: Predicting Model Transferability in a Self-challenging Fisher Space"
    from https://arxiv.org/abs/2207.03036
    """
    def __init__(self):
        self.logger = LoggerSetup("Metric [SFDA]").get_logger()
        self.logger.info("Booted: Metric [SFDA].")

    def __str__(self):
        return "EMMS"

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

    def initialize(self, embeddings: torch.Tensor, targets: torch.Tensor, bbox: torch.Tensor = None) -> None:
        self.logger.info("Initializing SFDA.")
        super().__init__("EMMS", embeddings, targets)
        self.bbox = bbox

        for i in range(len(self.class_labels)):
            self.logger.info(f"Class {self.class_labels[i]} has {self.class_label_counts[i]} samples.")

        self.logger.info("Initialization Complete.")

    def fit(self): 
        n = len(self.targets)
        num_classes = len(self.class_labels)
        
        X = self.embeddings
        y = self.targets

        SFDA_first = _SFDA()
        prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)
        
        prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True) 
        _, means_ = SFDA_first._class_means(X, y) 
        
        # ConfMix
        for y_ in range(num_classes):
            indices = np.where(y == y_)[0]
            y_prob = np.take(prob, indices, axis=0)
            y_prob = y_prob[:, y_]  
            X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                                (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]
        
        SFDA_second = _SFDA(shrinkage=SFDA_first.shrinkage)
        prob = SFDA_second.fit(X, y).predict_proba(X)   # n * num_cls

        # Expectation of p(y|x).
        sfda_score = np.sum(prob[np.arange(n), y]) / n
        self.logger.info(f"SFDA Score: {sfda_score}")
        return sfda_score