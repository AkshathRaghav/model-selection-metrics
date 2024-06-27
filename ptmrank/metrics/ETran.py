# ---
# Adapted from https://github.com/mgholamikn/ETran/blob/main/metrics.py
# ---

import torch 
import numpy as np 
from ..tools.lda import LDA 
from .Metric import Metric
from ..tools.logger import LoggerSetup

class ETran(Metric):
    """
    ETran Energy Score calculation as proposed in the ICCV 2023 paper
    "ETran: Energy-Based Transferability Estimation"
    from https://arxiv.org/abs/2308.02027
    """
    def __init__(self):
        self.logger = LoggerSetup("Metric [ETran]").get_logger()
        self.logger.info("Booted: Metric [ETran].")

    def __str__(self):
        return "ETran"

    def reset(self):
        self.embeddings = None
        self.targets = None
        self.class_labels = None
        self.class_label_counts = None 
    
    def test(self): 
        self.logger.info("Running test.")

        dim = 1024 
        bbox_dim = 4 
        embeddings = torch.rand(1000, dim)
        targets = torch.randint(0, 3, (1000,))
        bbox_targets = torch.rand(1000, bbox_dim)

        self.initialize(embeddings, targets)
        _ = self.fit() 
        self.reset() 

        self.initialize(embeddings, targets, bbox=bbox_targets)
        _ = self.fit()
        self.reset()

        self.logger.info("Success.")

    def initialize(self, embeddings: torch.Tensor, targets: torch.Tensor, bbox: torch.Tensor = None) -> None:
        self.logger.info("Initializing ETran.")
        super().__init__("ETran", embeddings, targets)
        self.bbox = bbox

        for i in range(len(self.class_labels)):
            self.logger.info(f"Class {self.class_labels[i]} has {self.class_label_counts[i]} samples.")

        self.logger.info("Initialization Complete.")

    @staticmethod
    def s_en(features): 
        """
        Deals with only embeddings

        - features: (num_samples, feature_dim)
        """
        energy_score=torch.logsumexp(features, dim=-1)
        chs = torch.argsort(energy_score)[0:5*len(energy_score)//1000]
        return energy_score[chs].mean() 

    @staticmethod
    def s_reg(features, targets):
        """
        Deals with both labels and bbox 

        - features: (num_samples, feature_dim)
        - targets: (num_samples, 1) or (num_samples, 4)
        """
        U, S, Vh = torch.linalg.svd(features, full_matrices=False)
        
        energy = torch.cumsum(S**2, dim=0) / torch.sum(S**2)
        rank = torch.searchsorted(energy, 0.8).item()
        
        S = S[:rank]
        U = U[:, :rank]
        Vh = Vh[:rank, :]
        
        S_inv = torch.diag(1.0 / S)
        features_pseudo_inv = Vh.T @ S_inv @ U.T
        bboxes_approximated = features @ features_pseudo_inv @ targets.float()
        
        return -torch.sum((targets - bboxes_approximated) ** 2) * (1/(targets.shape[0] * 4))
    
    @staticmethod
    def s_cls(features, labels):
        """
        Only deals with the classes 

        features (torch.Tensor): (num_samples, feature_dim).
        labels (torch.Tensor): (num_samples,).
        """

        n = len(labels)
        prob = LDA().fit(features, labels).predict_proba(features,labels)  
        prob = torch.from_numpy(prob).float() 
        return torch.sum(prob[torch.arange(n), labels]) / n
    
    def fit(self):
        s_en = ETran.s_en(self.embeddings)  
        self.logger.info(f"Energy Score: {s_en:.2f}")
        T = s_en

        s_cls = ETran.s_cls(self.embeddings, self.targets)
        self.logger.info(f"Classification Score: {s_cls:.2f}")
        T += s_cls

        if len(self.targets.shape) < 2: 
            self.targets = self.targets.unsqueeze(1)

        s_reg = ETran.s_reg(self.embeddings, self.targets)
        self.logger.info(f"Regression Score: {s_reg:.2f}")
        T += s_reg

        if self.bbox is not None: 
            s_reg = ETran.s_reg(self.embeddings, self.bbox)
            self.logger.info(f"Regression Score for bbox: {s_reg:.2f}")
            T += s_reg

        self.logger.info(f"T = {T:.2f}")
        return T
    
ETran().test()