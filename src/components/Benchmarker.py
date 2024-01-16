from .Model import Models
from .LogME import LogME
from .LEEP import LEEP
from .NCE import NCE
from .FeatureExtractors import (
    TextFeatureExtractor,
    ImageFeatureExtractor,
    AudioFeatureExtractor,
)
import os
from .Dataloaders import DataLoader
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import warnings
import logging

class Benchmarker:
    def __init__(
        self,
        models: Models,
        logme=True,
        ckpt=False,
        regression=False,
        auto_increment_if_failed=False,
        only_feature=False,
        leep=False,
        nce=False,
    ):
        self.models = models
        self.checkpoint = ckpt
        self.auto_increment_if_failed = auto_increment_if_failed
        self.only_feature = only_feature
        if int(logme) + int(leep) + int(nce) != 1:
            logger.error('Cannot have more than one of logme, leep, or nce as True!')
            raise ValueError("ERROR: Only one of logme, leep, or nce can be True")

        if logme:
            self.benchmark_model = LogME(regression=regression)
            self.regression = regression
            logging.critical(
                f'LogME initialized with{"" if regression else " no"} regression'
            )
        elif leep:
            self.benchmark_model = LEEP()
            logging.critical("LEEP initalized")
        else:
            self.benchmark_model = NCE()
            loggger.critical("NCE initalized")

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
            "Multimodal Not Supported": [
                "text-to-image",
                "image-to-text",
                "table-question-answering",
            ],
            AudioFeatureExtractor: [
                "automatic-speech-recognition",
                "audio-classification",
            ],
        }

    def __call__(self, design_specs):
        task = design_specs["task"]
        dataloader = design_specs["dataset"]
        if not isinstance(dataloader, DataLoader):
            raise ValueError("Dataset should be a `DataLoader` class!!")
        n = design_specs["n"]
        return self.benchmark_models(task, dataloader, n)

    def benchmark_models(self, task, dataloader, n=5):
        checkpoint_pth = f"../checkpoints/{task}_{str(dataloader)}"

        extractor_group = [
            x for x in self.extractor_groups if task in self.extractor_groups[x]
        ][0]
        if type(extractor_group) == str:
            logging.error(f'{extractor_group} not present in list!')
            raise ValueError(f"ERROR: {extractor_group}")

        logging.info(f"TASK: {task}")

        models = self.models(task, n)

        logging.debug(f"Models: {models}")

        if len(models) == 0:
            raise Error("ERROR: No models selected!")

        benchmarks = {}

        check = False
        for model_name, base_model_identifier, _, _ in models:
            if check and self.auto_increment_if_failed:
                models += self.models(task, 1)

            check = False

            logging.info("----------------------------------------")
            logging.info(f"Model_name: {model_name}")

            if self.checkpoint:
                model_checkpoint_pth = (
                    checkpoint_pth + f"/{model_name.replace('/', '_')}/"
                )
                try:
                    os.mkdir(model_checkpoint_pth)
                except:
                    pass

            if self.checkpoint: 
                use_checkpoint = False
                if os.path.exists(model_checkpoint_pth + "/config.json"):
                    try:
                        config = json.load(model_checkpoint_pth + "/config.json")
                        use_checkpoint = True
                    except:
                        use_checkpoint = False

            if not use_checkpoint:
                extractor = extractor_group(task, model_name)
                if not extractor.feature_extractor or not extractor.model:
                    logging.error(
                        f"Could not load {model_name}, aborting benchmark!"
                    )
                    check = True
                    continue
                else:
                    pass
            else:
                pass

            logging.critical("Extractor Loaded")

            if not use_checkpoint:
                try:
                    features, targets = self.extract_features_and_targets(
                        extractor, dataloader, n, only_feature=self.only_feature
                    )
                except ValueError as e:
                    logging.warning(e)
                    check = True
                    continue
            else:
                features, targets = config["features_pth"], config["targets_pth"]

            logging.critical("Feature Extraction Complete.")
            logging.info("Starting fitting the benchmarker.")

            score = 0

            if not check: 
                score = self.fit(features, targets)

                logging.critical(
                    f"{model_name} score: {score}."
                )
                logging.info("----------------------------------------")

            benchmarks[model_name] = score

            if self.checkpoint:
                np.save(model_checkpoint_pth + "/features.npy", features)
                np.save(model_checkpoint_pth + "/targets.npy", targets)
                config = {
                    "model_name": model_name,
                    "dataset_name": str(dataloader),
                    "score": score,
                    "features_pth": model_checkpoint_pth + "/features.npy",
                    "targets_pth": model_checkpoint_pth + "/targets.npy",
                }
                with open(model_checkpoint_pth + "/config.json", "w") as fp:
                    json.dump(config, fp)
            del use_checkpoint, features, targets
        return benchmarks

    def extract_features_and_targets(self, extractor, dataloader, n, only_feature=False):
        logging.info("Extracting Features and Targets")
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
            logging.error("ERROR: (Could be sampling_rate error for AudioModels)")
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

        logging.debug(f"Features Shape: {f.shape}")
        logging.debug(f"Labels Shape: {y.shape}")

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
