import os, json, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm

from .metrics.LogME import LogME
from .metrics.LEEP import LEEP
from .metrics.NCE import NCE
from .config import Config
from .models import ExhaustedModelsError
from .dataloaders import DataLoader
from .tools.logger import LoggerSetup
from .tools.feature_extractors import (
    TextFeatureExtractor,
    ImageFeatureExtractor,
    AudioFeatureExtractor,
)


class Benchmarker:
    def __init__(
        self,
        config: Config,
        regression=False
    ):
        logging_setup = LoggerSetup("Benchmarker")
        self.logger = logging_setup.get_logger()
        self.logger.info("Initializing Benchmarker")

        self.models = config.models
        self.persist_dir = config.persist_dir
        assert os.path.exists(self.persist_dir), f"Path {self.persist_dir} does not exist!"
        self.auto_increment_if_failed = config.auto_increment_if_failed
        self.only_feature = config.extract_features_only
        self.metrics = config.metrics


        self.extractor_groups = {
            TextFeatureExtractor: [
                "text-classification",
                "token-classification",
                "question-answering",
                "zero-shot-classification",
                "translation",
                "summarization",
                "conversational",
                "text-generation",
                "text2text-generation",
                "fill-mask",
                "sentence-similarity",
                "feature-extraction",
            ],
            ImageFeatureExtractor: [
                "image-classification",
                "object-detection",
                "image-segmentation",
                "video-classification",
            ],
            # "Multimodal Not Supported": [
            #     "text-to-image",
            #     "image-to-text",
            #     "table-question-answering",
            # ],
            AudioFeatureExtractor: [
                "automatic-speech-recognition",
                "audio-classification",
            ],
        }

    def __call__(self, task: str, dataloader: DataLoader, n: int):
        return self.benchmark_models(task, dataloader, n)

    def benchmark_models(self, task, dataloader, n=5):
        def create_dir(path):
            try:
                self.logger.info(f"Creating directory {path}")
                os.mkdir(path)
            except Exception as e:
                self.logger.error(e)
                pass

        
        self.logger.info("-" * 20)
        self.logger.info(f"Starting Benchmarking for {task} on {str(dataloader)}")
        self.logger.info(f"Number of models to benchmark: {n}, Total models available: {len(self.models)}")


        # Create a directory to store features if needed.
        checkpoint_pth = f"{self.persist_dir}/{task}_{str(dataloader)}"
        if self.persist_dir:
            create_dir(checkpoint_pth)

        # Choosing which extractor to use. 
        extractor_group = [
            x for x in self.extractor_groups if task in self.extractor_groups[x]
        ]
        if not extractor_group:
            self.logger.error(f'{extractor_group} not present in list!')
            raise ValueError(f"ERROR: {extractor_group}")
        else: 
            extractor_group = extractor_group[0]

        benchmarks = {}
        idx = 0

        while idx < n: 
            self.logger.info("-" * 10)

            # Get another model to benchmark.
            try: 
                model_name = self.models()
                self.logger.info(f"Model_name: {model_name}")
            except ExhaustedModelsError:
                self.logger.error("Exhausted all models. Stopping Benchmarker.")
                break
            
            # Skip if processed. 
            if model_name in benchmarks:
                continue

            # Store processed features if needed. 
            if self.persist_dir:
                model_checkpoint_pth = (
                    checkpoint_pth + f"/{model_name.replace('/', '_')}/"
                )
                try:
                    self.logger.info(f"Creating directory {model_checkpoint_pth}")
                    os.mkdir(model_checkpoint_pth)
                except:
                    self.logger.warning(f"Directory {model_checkpoint_pth} already exists.")
                    pass
                
                use_checkpoint = False
                if os.path.exists(os.path.join(model_checkpoint_pth, "/config.json")):
                    try:
                        config = json.load(model_checkpoint_pth + "/config.json")
                        use_checkpoint = True
                    except:
                        pass 

            if not use_checkpoint:
                extractor = extractor_group(task, model_name) # Loading feature extractor and task-specific AutoModel 
                if not extractor.feature_extractor or not extractor.model:
                    self.logger.error(
                        f"Could not load {model_name}, aborting benchmark!"
                    )
                    idx += self.increment_pass(False)
                    continue
                else: 
                    self.logger.critical("Extractor Loaded")

                try:
                    features, targets = self.extract_features_and_targets(
                        extractor, dataloader, n, only_feature=self.only_feature
                    )
                except ValueError as e:
                    self.logger.warning(e)
                    idx += self.increment_pass(False)
                    continue
            else:
                features, targets = np.load(config["features_pth"]), np.load(config["targets_pth"])

            self.logger.critical("Feature Extraction Complete.")
            self.logger.info("Starting fitting the benchmarker.")

            scores = {}

            if not self.only_feature:
                for metric in self.metrics:
                    self.logger.info(f"Running {metric} metric.")
                    score = metric(features, targets)
                    self.logger.critical(
                        f"{model_name} {metric} score: {score}."
                    )
                    scores[str(metric)] = score
                    self.logger.info("----------------------------------------")

            benchmarks[model_name] = scores

            if self.persist_dir:
                np.save(model_checkpoint_pth + "/features.npy", features)
                np.save(model_checkpoint_pth + "/targets.npy", targets)
                config = {
                    "model_name": model_name,
                    "dataset_name": str(dataloader),
                    "score": benchmarks[model_name],
                    "features_pth": model_checkpoint_pth + "/features.npy",
                    "targets_pth": model_checkpoint_pth + "/targets.npy",
                }
                with open(model_checkpoint_pth + "/config.json", "w") as fp:
                    json.dump(config, fp)

            del use_checkpoint, features, targets

            idx += self.increment_pass(True)

        self.logger.info("-" * 20)
        return benchmarks
    
    def increment_pass(self, success: bool): 
        if success: return 1 
        elif not success and not self.auto_increment_if_failed: return 1
        elif not success and self.auto_increment_if_failed: return 0

    def extract_features_and_targets(self, extractor, dataloader, n, only_feature=False):
        self.logger.info("Extracting Features and Targets")

        dataloader.reset()
        features_list = []
        labels_list = []

        progress_bar = tqdm(total=dataloader.max_idx, desc="       Progress: ")

        batch = dataloader()
        while batch:
            features, labels = extractor(batch, only_feature=only_feature)

            if not len(features) and not len(labels):
                batch = dataloader()
                continue

            features_list.append(features)
            if type(labels) != np.ndarray:
                labels = np.array([labels])
            labels_list.append(labels)

            progress_bar.update(1)

            batch = dataloader()

        progress_bar.close()

        if len(features_list) < n // 2:
            self.logger.error("ERROR: (Could be sampling_rate error for AudioModels)")
            raise ValueError("ERROR: Not enough features extracted!.")

        f = np.concatenate(features_list, axis=0)
        y = np.concatenate(
            [np.array(self.convert_labels_with_pandas(labels_list))], axis=0
        )

        f = f.reshape(f.shape[0], -1)

        if not self.regression:
            y = y.astype(int)
        else:
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)

        self.logger.debug(f"Features Shape: {f.shape}")
        self.logger.debug(f"Labels Shape: {y.shape}")

        return f, y

    def convert_labels_with_pandas(self, labels_list):
        flat_labels = [label for sublist in labels_list for label in sublist]
        labels_series = pd.Series(flat_labels)
        labels_series = labels_series.astype("category").cat.codes
        return labels_series.values

    def fit(self, features, labels):
        warnings.filterwarnings("ignore")
        self.benchmark_model.reset()
        return self.benchmark_model.fit(features, labels)

    def is_regression(self):
        if all(isinstance(label, (int, float)) for label in self.labels):
            unique_labels = set(labels)
            if len(unique_labels) > len(labels) * 0.1:
                return True

            if all(isinstance(label, int) for label in labels):
                return False

        return False
