# --- 
# Adapted from: https://github.com/thuml/LogME
# ---

import numpy as np
from numba import njit
from .Metric import Metric
from ..tools.logger import LoggerSetup

@njit
def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m

@njit
def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh

class LogME(Metric):
    """ 
    LogME calculation as proposed in the ICML 2021 and JMLR 2022 papers
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        from http://proceedings.mlr.press/v139/you21b.html
        "Ranking and Tuning Pre-trained Models: A New Paradigm for Exploiting Model Hubs"
        from https://arxiv.org/abs/2110.10545

    Note: 
        Lower LogME scores reflects inability to fit the embeddings to the targets. Closer to 0 the better
    """
    def __init__(self): 
        self.logger = LoggerSetup("Metric [LogME]").get_logger()
        self.logger.info("Booted: Metric [LogME].")

    def __str__(self):
        return "LogME"
    
    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def test(self): 
        self.logger.info("Running test.")

        dim = 1024
        embeddings = np.random.rand(1000, dim)  
        targets = np.random.randint(0, 3, 1000)

        self.initialize(embeddings, targets)
        _ = self.fit(embeddings, targets)
        self.reset()

        self.logger.info("Success.")

    def replicate_paper_results(self):
        """ 
        Similar to GBC, LogME results are evaluated using Kendall's Tau, which requires more samples and more metrics. To avoid this, we'll be replicating the theoretical guarantee of LogME. 

        The following code will replicate Figure 3 from https://arxiv.org/pdf/2102.11005. It'll run synthetically generated samples, with varying levels of noise, and plot the LogME score against the standard deviation of the noise. As the noise increases (distance between clusters decreases), LogME decreases. 
        
        The results are consistent with the paper. 
        """
        import matplotlib.pyplot as plt

        def generate_separable_embeddings(n_samples, dim, n_classes):
            embeddings = []
            targets = []
            
            for i in range(n_classes):
                center = np.random.rand(dim) * 20  # Spread the class centers far apart
                class_embeddings = np.random.randn(n_samples // n_classes, dim) + center
                class_targets = np.full(n_samples // n_classes, i)
                
                embeddings.append(class_embeddings)
                targets.append(class_targets)
                
            embeddings = np.vstack(embeddings)
            targets = np.concatenate(targets)
            
            return embeddings, targets

        def generate_linear_embeddings_and_targets(n_samples, dim, n_targets):
            embeddings = np.linspace(0, 100, n_samples).reshape(-1, 1)  # [N, 1]
            embeddings = np.hstack([embeddings for _ in range(dim)])  # [N, dim]

            targets = np.linspace(0, 100, n_samples).reshape(-1, 1)  # [N, 1]
            targets = np.hstack([targets for _ in range(n_targets)])  # [N, n_targets]

            return embeddings, targets

        def add_noise(embeddings, noise_level):
            noise = np.random.randn(*embeddings.shape) * noise_level
            noisy_embeddings = embeddings + noise
            return noisy_embeddings

        def visualize_embeddings(embeddings, targets, title):
            import umap
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
            
            reducer = umap.UMAP()
            embedding_2d = reducer.fit_transform(embeddings)
            
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=targets[:, 0], cmap='Spectral', s=5)
            plt.colorbar(scatter, label='Target Value')
            plt.title(title)
            plt.xlabel('UMAP 1')
            plt.ylabel('UMAP 2')
            plt.grid(True)
            plt.show()

        dim = 1024
        n_samples = 1000
        n_classes = 3
        std_range = range(26)

        # W/O Regression 
        embeddings, targets = generate_separable_embeddings(n_samples, dim, n_classes)
        logme_values = []

        for std in std_range:
            noisy_embeddings = add_noise(embeddings, std)
            self.initialize(noisy_embeddings, targets)
            score = self.fit()
            self.reset()
            logme_values.append(score)

        plt.plot(std_range, logme_values, marker='o')
        plt.xlabel('standard deviation of noise')
        plt.ylabel('LogME')
        plt.grid(True)
        plt.show()

        dim = 1024
        n_samples = 1000
        n_targets = 10
        std_range = range(15)

        # W/ Regression 
        embeddings, targets = generate_linear_embeddings_and_targets(n_samples, dim, n_targets)
        logme_values = []

        for std in std_range:
            noisy_embeddings = add_noise(embeddings, std)
            self.initialize(noisy_embeddings, targets, regression=True)
            score = self.fit()
            self.reset()
            logme_values.append(score)
            # visualize_embeddings(embeddings, targets, f"Regression: {std}")
            
        plt.plot(std_range, logme_values, marker='o')
        plt.xlabel('standard deviation of noise')
        plt.ylabel('LogME')
        plt.grid(True)
        plt.show()

    def initialize(self, embeddings: np.ndarray, targets: np.ndarray, regression: bool = False) -> None:
        super().__init__("LogME", embeddings, targets, False)

        self.logger.info(f"Embeddings shape: {self.embeddings.shape}; Targets shape: {self.targets.shape}; Regression: {regression}")
        self.regression = regression
        self.fitted = False
        self.reset()
        self.logger.info("Initialization Complete.")

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        N, D = f.shape  # k = min(N, D)
        if N > D: # direct SVD may be expensive
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k
        # s.shape = k
        # vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        sigma_full_size = sigma
        if N < D:  # pad sigma to size D
            sigma_full_size = np.pad(sigma, ((0, D - N), (0, 0)), 'constant')

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)

        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            x = u.T @ y_  # x has shape [k, 1], but actually x should have shape [N, 1]
            x2 = x ** 2
            res_x2 = (y_ ** 2).sum() - x2.sum()  # if k < N, we compute sum of xi for 0 singular values directly

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma_full_size)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma_full_size)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)

        self.ms = np.stack(self.ms)
        
        return np.mean(evidences)

    def fit(self):
        """
        f: [N, F], feature matrix from pre-trained model
        y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well f can fit y directly)
        """
        f = self.embeddings
        y = self.targets

        if self.fitted:
            self.logger.warning("Already fitted, reset before fitting again.")
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        score = self._fit_fixed_point(f, y)

        self.logger.info(f"LogME score: {score}")
        return score

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        return np.argmax(logits, axis=-1)


# LogME().replicate_paper_results()