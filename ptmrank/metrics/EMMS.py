import torch
import torch.nn.functional as F
from .Metric import Metric
from ..tools.logger import LoggerSetup
from ..tools.flabel import FLabel

class EMMS(Metric):
    """
    EMMS Score calculation as proposed in 
    "Foundation Model is Efficient Multimodal Multitask Model Selector"
    from https://arxiv.org/abs/2308.06262
    """
    def __init__(self):
        self.logger = LoggerSetup("Metric [EMMS]").get_logger()
        self.logger.info("Booted: Metric [EMMS].")

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
        self.logger.info("Initializing EMMS.")
        super().__init__("EMMS", embeddings, targets)
        self.bbox = bbox

        for i in range(len(self.class_labels)):
            self.logger.info(f"Class {self.class_labels[i]} has {self.class_label_counts[i]} samples.")

        self.logger.info("Initialization Complete.")

    def fit(self) -> float:
        """ 
        Modelling relationship between model features and the label embeddings. f_y (F-Labels) generated using tools::flabel.py. 
        """ 

        x = self.embeddings
        y = FLabel(self.targets, k=["clip", "bert", "gpt-2"])

        N, D2, K = y.shape

        y_mean = torch.mean(y, dim=0, keepdim=True)
        y_std = torch.std(y, dim=0, keepdim=True)
        epsilon = 1e-8
        y = (y - y_mean) / (y_std + epsilon)

        x_mean = torch.mean(x, dim=0)
        x_std = torch.std(x, dim=0)
        x = (x - x_mean) / (x_std + epsilon)

        lam = torch.full((K,), 1/K, dtype=torch.float32)
        T = torch.matmul(y, lam)

        for k in range(1):
            w = torch.linalg.lstsq(x, T).solution
            w1 = w

            y_flat = y.reshape(N * D2, K)
            x_w = torch.matmul(x, w).reshape(N * D2)
            lam = torch.linalg.lstsq(y_flat, x_w).solution
            lam = F.softmax(lam, dim=0)

            T = torch.matmul(y, lam).reshape(N, D2)

        y_pred = torch.matmul(x, w1)
        res = torch.sum((y_pred - T) ** 2) / (N * D2)

        return -res.item()

EMMS().test() 