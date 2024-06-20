from typing import Any, Union
from sklearn.decomposition import PCA
from scipy.stats import kendalltau 
import torch 
import numpy as np 
from ..tools.cov import cov

class MetricError(Exception):
    pass

class Metric: 
    def __init__(self, name: str, embeddings: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor], structured: bool = True) -> None: 
        self.name = name

        if isinstance(embeddings, np.ndarray): embeddings = torch.from_numpy(embeddings)
        if isinstance(targets, np.ndarray): targets = torch.from_numpy(targets)

        self.embeddings = embeddings
        self.targets = targets

        if structured:
            self.class_labels, self.class_label_counts = torch.unique(targets, return_counts=True)

    def __str__(self): 
        return self.name
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.fit() 
    
    def fit(self): 
        raise NotImplementedError

    def replicate_paper_results(self): 
        """ 
        Only for modules that have not been adapted from the original source.
        """
        raise NotImplementedError

    def apply_PCA(self, embeddings: torch.Tensor, n_components: int = 64): 
        return PCA(n_components=n_components).fit_transform(embeddings)
    
    def apply_kendall_tau(self, accuracies: torch.Tensor, scores: torch.Tensor):
        return kendalltau(accuracies, scores).statistic
    
    def scale(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings / torch.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)

    def check_positive_semidefinite(matrix: torch.Tensor) -> bool:
        return torch.all(torch.linalg.eigvals(matrix) > 0)

    def check_cond(matrix: torch.Tensor) -> bool:
        return torch.linalg.cond(matrix) < 10
    
    def _mu(self, means: torch.Tensor) -> torch.Tensor: 
        for ind, c in enumerate(self.class_labels):
            means[ind] = torch.mean(self.embeddings[self.targets == c, :], dim=(0,))
        return means

    def _cov(self, covariances: torch.Tensor) -> torch.Tensor: 
        for ind, c in enumerate(self.class_labels):
            class_embeddings = self.embeddings[self.targets == c, :]
            if class_embeddings.shape[0] > 1:
                reg_constant = 1e-5  # Small regularization constant
                variances = cov(class_embeddings, rowvar=False)
                variances += reg_constant * torch.eye(variances.shape[0])
                covariances[ind][:, :] = variances
            else:
                # If only one example in the class, avoid singular matrix by adding small identity
                covariances[ind][:, :] = torch.eye(self.embeddings.shape[1]) * 0.01
            
            # # assert check_cond(covariances[c]), f"Condition number of covariance matrix is too high."
            # # assert check_positive_semidefinite(covariances[c]), f"Covariance matrix is not positive semidefinite."

        return covariances

class MetricNP: 
    def __init__(self, name: str, embeddings: np.ndarray, targets: np.ndarray, structured: bool = True) -> None: 
        self.name = name
        self.embeddings = embeddings
        self.targets = targets

        if structured:
            assert self.check_structured_classes(), f"{self.name} requires structured class labels."
            self.class_labels = np.unique(targets).astype(np.int64)

    def __str__(self): 
        return self.name
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.fit() 
    
    def fit(self): 
        raise NotImplementedError

    def check_structured_classes(self) -> bool:
        """ 
        Checks if the targets are structured class labels. 
        Transferrability metrics do not accoun for unstructured outputs from generative models like LLMs/VAEs/GANs/etc.
        """

        if isinstance(self.targets, np.ndarray):
            return len(np.unique(self.targets)) != self.targets.shape[0]
        else: 
            raise ValueError("Targets must be either a numpy array.")

    def replicate_paper_results(self): 
        """ 
        Only for modules that have not been adapted from the original source.
        """
        raise NotImplementedError

    def apply_PCA(self, embeddings: np.ndarray, n_components: int = 64): 
        return PCA(n_components=n_components).fit_transform(embeddings)
    
    def apply_kendall_tau(self, accuracies: np.ndarray, scores: np.ndarray):
        return kendalltau(accuracies, scores).statistic
    
    def scale(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings / np.linalg.norm(embeddings, ord=2, axis=1, keepdims=True)