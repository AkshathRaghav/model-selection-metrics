# --- 
# Adapted from: https://github.com/thuml/LogME
# ---

import torch 
from .Metric import Metric
from ..tools.logger import LoggerSetup

class LEEP(Metric):
    """
    LEEP calculation as proposed in the ICML 2020 paper
    "LEEP: A New Measure to Evaluate Transferability of Learned Representations"
    from http://proceedings.mlr.press/v119/nguyen20b/nguyen20b-supp.pdf
    """
    def __init__(self):
        self.logger = LoggerSetup("Metric [LEEP]").get_logger()
        self.logger.info("Booted: Metric [LEEP].")

    def __str__(self):
        return "LEEP"
    
    def test(self): 
        self.logger.info("Running test.")

        dim = 1024
        pseudo_source_label = torch.rand(1000, dim)
        target_label = torch.randint(0, 3, (1000,))

        self.initialize(pseudo_source_label, target_label)
        _ = self.fit()
        
        self.logger.info("Success.")

    def initialize(self, pseudo_source_label, target_label) -> None:
        self.logger.info("Initializing LEEP.")

        super().__init__("LEEP", None, pseudo_source_label)
        self.transfer_targets = target_label

        self.logger.info("Initialization Complete.")
    
    def fit(self): 
        """ 
        self.targets: shape [N, C_s]; from source
        self.transfer_targets: shape [N], elements in [0, C_t); from target
        """
        N, C_s = self.targets.shape
        target_label = self.transfer_targets.reshape(-1)
        C_t = self.transfer_targets.shape[0]   # the number of target classes
        normalized_prob = self.targets / float(N)  # sum(normalized_prob) = 1
        joint = torch.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)

        for i in range(C_t):
            this_class = normalized_prob[target_label == i]
            row = torch.sum(this_class, dim=0)
            joint[i] = row
        p_target_given_source = (joint / torch.sum(joint, dim=0, keepdims=True)).T  # P(y | z)

        empirical_prediction = self.targets @ p_target_given_source.float()
        empirical_prob = torch.Tensor([predict[label] for predict, label in zip(empirical_prediction, target_label)])
        leep_score = torch.mean(torch.log(empirical_prob))

        self.logger.info(f"LEEP Score: {leep_score}")
        return leep_score

LEEP().test()