from ptmrank.metrics.Metric import Metric, MetricError
from ptmrank.tools.logger import LoggerSetup
import torch 
from typing import Union 
import numpy as np 

class GBC(Metric):
    """ 
    GBC calculation as proposed in the CVPR 2022 Paper 
    "Transferability Estimation using Bhattacharyya Class Separability"
    from https://arxiv.org/pdf/2111.12780

    Note: 
        More negative the GBC is, the worse the potential transfer accuracy gets.
    """
    
    def __init__(self): 
        self.logger = LoggerSetup("Metric [GBC]").get_logger()
        self.logger.info("Initializing Metric [GBC].")

    def __str__(self):
        return "Gaussian Bhattacharyya Coefficient"
    
    def reset(self):
        self.means = None
        self.covariances = None
        self.embeddings = None
        self.targets = None
        self.class_labels = None
        self.class_labels_counts = None 
        
    def test(self): 
        self.logger.info("Running test.")

        dim = 1024
        embeddings = torch.rand(1000, dim)  
        targets = torch.randint(0, 5, (1000,))

        self.initialize(embeddings, targets)
        _ = self.fit()
        self.reset()

        self.logger.info("Success.")

    def replicate_paper_results(self): 
        """ 
        Results in the paper are defined using the weighted Kendall Tau Correlation Coefficient. Check self.apply_kendall_tau() for more information.

        Instead, we'll focus on Figure 4, to show prove the GBC values and plots for the CIFAR-10 subset. Since we don't have the exact setup, dataset split, parameters and weights, and are loading weights from HF, we'll just replicate the results as semantically closely as possible.

        Compare the scores with the plots. More overlap in the clusters, the worse the transfer accuracy will be. 
        """
        def convert_to_rgb(data: np.ndarray) -> np.ndarray:
            return np.stack([data[0:1024].reshape(32, 32), data[1024:2048].reshape(32, 32), data[2048:3072].reshape(32, 32)], axis=-1)
        
        def save(type: str, embeddings: np.ndarray, targets: np.ndarray): 
            with open(rf"D:\model-selection-metrics\ptmrank\metrics\embeddings_{type}.npy", 'wb') as f: 
                np.save(f, embeddings)
            with open(rf"D:\model-selection-metrics\ptmrank\metrics\targets_{type}.npy", 'wb') as f: 
                np.save(f ,targets)

        def visualize(model: str, data: np.ndarray, labels: np.ndarray): 
            import umap
            import matplotlib.pyplot as plt
            from scipy.spatial import ConvexHull
            from sklearn.cluster import KMeans

            reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean', random_state=42)
            embedding = reducer.fit_transform(data)

            plt.figure(figsize=(12, 10))
            unique_labels = np.unique(labels)
            colors = plt.cm.get_cmap('viridis', len(unique_labels))

            for i, label in enumerate(unique_labels):
                points = embedding[labels == label]
                plt.scatter(points[:, 0], points[:, 1], color=colors(i), label=f'Class {label}', alpha=0.6)

                if points.shape[0] > 10:  
                    kmeans = KMeans(n_clusters=1)  
                    kmeans.fit(points)
                    cluster_center = kmeans.cluster_centers_[0]
                    distances = np.linalg.norm(points - cluster_center, axis=1)
                    central_points = points[distances < np.percentile(distances, 75)]  

                    if central_points.shape[0] > 2:
                        hull = ConvexHull(central_points)
                        plt.fill(central_points[hull.vertices, 0], central_points[hull.vertices, 1],
                                color=colors(i), alpha=0.5, edgecolor='none')

            plt.title(f'Feature distribution of CIFAR-10 for {model}')
            plt.legend(title='Classes')
            plt.grid(True)
            plt.show()


        self.logger.info("Replicating Paper Results for GBC Metric.")
        self.logger.info("Loading CIFAR-10 Subset.")

        # import pickle 

        # with open(r"D:\model-selection-metrics\assets\cifar_10\data_batch_1", 'rb') as fo:
        #     data = pickle.load(fo, encoding='bytes')
        
        # labels = data[b'labels']
        # images = data[b'data']

        # del data 

        self.logger.info("Loading MobileNet Model and Processor.")  

        # from transformers import AutoImageProcessor, MobileNetV2Model
        # import torch

        # processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
        # model = MobileNetV2Model.from_pretrained("google/mobilenet_v2_1.0_224")  
        
        self.logger.info("Model and Processor Loaded.")

        self.logger.info("Extracting Features from CIFAR-10 images through MobileNet.")

        n = 5000

        # embeddings = np.zeros((n, 1280), dtype=np.float64)
        # for cnt, image in enumerate(images[:n]):
        #     with torch.no_grad():
        #         embeddings[cnt] = model(**processor(images=convert_to_rgb(image), return_tensors="pt")).pooler_output[0]
        # targets = np.array(labels[:n], dtype=np.float64)

        # save("mobilenet", embeddings, targets)

        with open(r"ptmrank/metrics/embeddings_mobilenet.npy", 'rb') as f: 
            embeddings = np.load(f)
        with open(r"ptmrank/metrics/targets_mobilenet.npy", 'rb') as f: 
            targets = np.load(f)
                    
        self.logger.info(f"Features Extracted with shape: {embeddings.shape}")

        self.initialize(embeddings, targets)
        _ = self.fit()
        self.reset() 

        # import timm
        # from PIL import Image

        self.logger.info("Loading DenseNet Model and Processor.")  

        # model = timm.create_model('densenet121.ra_in1k', num_classes=0, pretrained=True)
        # model = model.eval()
        # data_config = timm.data.resolve_model_data_config(model)
        # transforms = timm.data.create_transform(**data_config, is_training=False)

        self.logger.info("Model and Processor Loaded.")

        self.logger.info("Extracting Features from CIFAR-10 images through DenseNet.")

        # embeddings = np.zeros((n, 1024), dtype=np.float64)
        # for cnt, image in enumerate(images[:n]):
        #     with torch.no_grad():
        #         embeddings[cnt] = model(transforms(Image.fromarray(convert_to_rgb(image))).unsqueeze(0))
        # targets = np.array(labels[:n], dtype=np.float64)

        # save("densenet", embeddings, targets)

        with open(r"ptmrank/metrics/embeddings_densenet.npy", 'rb') as f: 
            embeddings = np.load(f)
        with open(r"ptmrank/metrics/targets_densenet.npy", 'rb') as f: 
            targets = np.load(f)

        self.logger.info(f"Features Extracted with shape: {embeddings.shape}")

        self.initialize(embeddings, targets)
        _ = self.fit()
        self.reset() 

    def initialize(self, embeddings: Union[np.ndarray, torch.Tensor], targets: Union[np.ndarray, torch.Tensor]) -> None:
        super().__init__("Gaussian_Bhattacharyya_Coefficient", embeddings, targets)

        self.dim = 64 

        unique = self.class_labels
        counts = self.class_label_counts

        for i in range(len(unique)):
            self.logger.info(f"Class {unique[i]} has {counts[i]} samples.")

            if counts[i] < 65: 
                self.logger.warning(f"Class {counts[i]} has less than 64 samples.")
                self.logger.critical(f"Setting n_dim to 32.")
                self.dim = 32
            elif counts[i] < 33: 
                self.logger.warning(f"Class {counts[i]} has less than 32 samples.")
                self.logger.error(f"Aborting. GBC requires at least 32 samples per class.")
                raise MetricError(f"Class {counts[i]} has less than 32 samples.")
            
        if self.dim == 64:
                self.logger.info(f"Setting n_dim to 64.")

        self.logger.info("Applying PCA to Embeddings.")
        self.embeddings = torch.from_numpy(self.apply_PCA(embeddings, n_components=self.dim))
        self.logger.info("PCA Applied.")

        self.logger.info(f"Shape of final featurs: {self.embeddings.shape}")

        self.means = torch.zeros((len(self.class_labels), self.dim), dtype=torch.float64)
        self.covariances = torch.zeros((len(self.class_labels), self.dim, self.dim), dtype=torch.float64)

        self.logger.info("Initialization Complete.")

    @staticmethod
    def _d_b(mean1, cov1, mean2, cov2):
        cov1 = torch.diag(torch.diag(cov1))
        cov2 = torch.diag(torch.diag(cov2))

        cov_mean = (cov1 + cov2) / 2
        inv_cov_mean = torch.linalg.pinv(cov_mean)
        det_cov_mean = torch.linalg.det(cov_mean)
        det_cov1 = torch.linalg.det(cov1)
        det_cov2 = torch.linalg.det(cov2)

        det_ratio = det_cov_mean / torch.sqrt(det_cov1 * det_cov2) 

        return inv_cov_mean, det_ratio
    
    def gbc(self):
        n_classes = len(self.means)

        gbc_score = 0.0
        for i in range(n_classes):
            for j in range(n_classes):
                if i == j: continue
                inv_cov_mean, det_ratio = GBC._d_b(None, self.covariances[i], None, self.covariances[j])
                mean_diff = self.means[i] - self.means[j]
                term1 = 0.125 * torch.matmul(mean_diff, torch.matmul(inv_cov_mean, mean_diff))
                term2 = 0.5 * torch.log(det_ratio)
                term = torch.exp(-(term1 + term2))
                gbc_score += term

        return -gbc_score    

    def fit(self): 
        self.logger.info("Fitting GBC Metric.")

        self.means = self._mu(self.means)
        self.covariances = self._cov(self.covariances)
        self.logger.info("Means and Covariances Calulated.")

        score = self.gbc() 
        self.logger.info(f"GBC Score: {score:.2f}")

        return score
    
GBC().replicate_paper_results()

# python -m ptmrank.metrics.GBC_torch
