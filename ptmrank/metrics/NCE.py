# --- 
# Adapted from: https://github.com/thuml/LogME
# ---


import numpy as np
from .Metric import Metric
from ..tools.logger import LoggerSetup

class NCE(Metric):
    """
    NCE calculation as proposed in the ICCV 2019 paper
    "Transferability and Hardness of Supervised Classification Tasks"
    from https://arxiv.org/abs/1908.08142

    :param source_label: shape [N], elements in [0, C_s), often got from taken argmax from pre-trained predictions
    :param target_label: shape [N], elements in [0, C_t)

    Note: 
        Returns the negative conditional entropy between source and target labels.  
    """
    def __init__(self):
        self.logger = LoggerSetup("Metric [NCE]").get_logger()
        self.logger.info("Booted: Metric [NCE].")

    def __str__(self):
        return "NCE"
    
    def test(self): 
        self.logger.info("Running test.")

        pseudo_source_label = np.random.randint(0, 3, 1000)
        target_label = np.random.randint(0, 3, 1000)

        self.initialize(pseudo_source_label, target_label)
        _ = self.fit()
        
        self.logger.info("Success.")

    def initialize(self, source_label, target_label) -> None:
        self.logger.info("Initializing NCE.")

        super().__init__("NCE", None, source_label)
        self.transfer_targets = target_label

        self.logger.info("Initialization Complete.")

    def fit(self):
        """
        self.targets: shape [N, C_s]; from source
        self.transfer_targets: shape [N], elements in [0, C_t); from target
        """
        C_t = int(np.max(self.transfer_targets) + 1)  # the number of target classes
        C_s = int(np.max(self.targets) + 1)  # the number of source classes
        N = len(self.targets)
        joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for the joint distribution, shape [C_t, C_s]
        for s, t in zip(self.targets, self.transfer_targets):
            s = int(s)
            t = int(t)
            joint[t, s] += 1.0 / N
        p_z = joint.sum(axis=0, keepdims=True)  # shape [1, C_s]
        p_target_given_source = (joint / p_z).T  # P(y | z), shape [C_s, C_t]
        mask = p_z.reshape(-1) != 0  # valid Z, shape [C_s]
        p_target_given_source = p_target_given_source[mask] + 1e-20  # remove NaN where p(z) = 0, add 1e-20 to avoid log (0)
        entropy_y_given_z = np.sum(- p_target_given_source * np.log(p_target_given_source), axis=1, keepdims=True)  # shape [C_s, 1]
        conditional_entropy = np.sum(entropy_y_given_z * p_z.reshape((-1, 1))[mask]) # scalar
        self.logger.info(f"NCE Score: {- conditional_entropy}")   
        return - conditional_entropy

NCE().test()