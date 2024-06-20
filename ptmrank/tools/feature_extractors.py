from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoFeatureExtractor,
    AutoProcessor,
)

import torch
import numpy as np 
import warnings
from typing import Union, Any, List

from .logger import LoggerSetup

class Embedder: 
    def __init__(self) -> None:
        self.logger = LoggerSetup("Embedder").get_logger()
        self.logger.info("Initializing Embedder.")
        self.embedder = None 
        warnings.simplefilter("error")

    def generate(self, input: Union[torch.Tensor, np.ndarray]) -> Any:
        self.logger.info("Starting embedding generation.")
        if self.embedder is None:
            raise ValueError("Embedder not initialized.")
        return self.embedder(input)

    def load(self, model_name, options): 
        for option in options: 
            try: 
                return option.from_pretrained(model_name, trust_remote_code=True)
            except: 
                pass
        return None

    def psuedo_pooler_output(self, output, key="hidden_states"): 
        assert hasattr(output, key), f"`{key}` not found in output of type {type(output)}"
        assert len(output[key][-1].shape) == 3, f"`{key}` shape {output.hidden_states[-1].shape} not acceptable for embedding."
        return output[key][-1].mean(dim=1)
    
    def check_embedding(self, embedding): 
        embedding = embedding.squeeze() if len(embedding.shape) > 2 else embedding
        assert len(embedding.shape) == 2, f"Vector shape {embedding.shape} not acceptable for embedding."
        return embedding

class TextFeatureExtractor:
    def __init__(self, task_name: str, model_name: str):
        super().__init__()

        self.feature_extractor = self.load_feature_extractor(model_name)
        self.logger.info(f"Feature Extractor loaded")
        self.model = self.load_model(model_name, task_name)
        self.logger.info(f"Model loaded")

    def load_feature_extractor(self, model_name):
        return self.load(model_name, [AutoTokenizer, AutoFeatureExtractor])

    def load_model(self, task_name, model_name):
        from transformers import (
            AutoModelForCausalLM,
            AutoModelForMaskedLM,
            AutoModelForSeq2SeqLM,
            AutoModelForSequenceClassification,
            AutoModelForTokenClassification,
            AutoModelForQuestionAnswering,
            AutoModelForTextEncoding,
        )
        classes = {
            "text-classification": [
                AutoModelForSequenceClassification,
            ],
            "token-classification": [
                AutoModelForTokenClassification,
            ],
            "question-answering": [
                AutoModelForQuestionAnswering,
            ],
            "zero-shot-classification": [
                AutoModelForSequenceClassification,
            ],
            "translation": [
                AutoModelForSeq2SeqLM,
            ],
            "summarization": [
                AutoModelForSeq2SeqLM,
            ],
            "conversational": [
                AutoModelForCausalLM,
            ],
            "text-generation": [
                AutoModelForCausalLM,
            ],
            "text2text-generation": [
                AutoModelForSeq2SeqLM,
            ],
            "fill-mask": [
                AutoModelForMaskedLM,
            ],
            "sentence-similarity": [
                AutoModelForSequenceClassification,
            ],
            "feature-extraction": [
                AutoModelForTextEncoding,
            ],
        }

        return self.load(model_name, classes.get(task_name, [AutoModel]))

    def __call__(self, text: str):
        with torch.no_grad():
            output = self.model.generate(self.tokenizer.encode(text, return_tensors="pt", padding=True, truncation=True), output_scores=True, return_dict_in_generate=True)    
            embedding = self.psuedo_pooler_output(output)

        return self.check_embedding(embedding)

class ImageEmbedder(Embedder):
    def __init__(self, task_name: str, model_name: str):
        super().__init__()

        self.feature_extractor = self.load_feature_extractor(model_name)
        self.logger.info(f"Feature Extractor loaded")
        self.model = self.load_model(model_name, task_name)
        self.logger.info(f"Model loaded")
    
    def load_feature_extractor(self, model_name: str):
        from transformers import AutoImageProcessor
        return self.load(model_name, [AutoImageProcessor, AutoFeatureExtractor])

    def load_model(self, model_name: str, task_name="auto-model"):
        from transformers import (
            AutoModelForDepthEstimation,
            AutoModelForImageClassification,
            AutoModelForVideoClassification,
            AutoModelForMaskedImageModeling,
            AutoModelForObjectDetection,
            AutoModelForImageSegmentation,
            AutoModelForImageToImage,
            AutoModelForSemanticSegmentation,
            AutoModelForInstanceSegmentation,
            AutoModelForUniversalSegmentation,
            AutoModelForZeroShotImageClassification,
            AutoModelForZeroShotObjectDetection,
        )
        classes = {
            "depth-estimation": [AutoModelForDepthEstimation],
            "image-classification": [AutoModelForImageClassification],
            "video-classification": [AutoModelForVideoClassification],
            "object-detection": [AutoModelForObjectDetection],
            "image-segmentation": [
                AutoModelForImageSegmentation,
                AutoModelForUniversalSegmentation,
                AutoModelForSemanticSegmentation,
                AutoModelForInstanceSegmentation,
            ],
            "zero-shot-image-classification": [AutoModelForZeroShotImageClassification],
        }

        return self.load(model_name, classes.get(task_name, [AutoModel]))

    def __call__(self, images: Union[torch.Tensor, np.ndarray], batch_size: int = 1):
        with torch.no_grad():
            output = self.model(**self.feature_extractor(images, return_tensors="pt"))
            if hasattr(output, "pooler_output"): 
                embedding = output.pooler_output
            else: 
                embedding = self.psuedo_pooler_output(output)

        return self.check_embedding(embedding)

class AudioEmbedder(Embedder):
    def __init__(self, task_name: str, model_name: str):
        super().__init__()

        self.feature_extractor = self.load_feature_extractor(model_name)
        self.logger.info(f"Feature Extractor loaded")
        self.model = self.load_model(task_name, model_name)
        self.logger.info(f"Model loaded")

    def load_feature_extractor(self, model_name):
        return self.load(model_name, [AutoFeatureExtractor, AutoProcessor])

    def load_model(self, task_name, model_name):
        from transformers import AutoModelForAudioClassification
        return self.load(model_name, [AutoModelForAudioClassification, AutoModel])

    def __call__(self, audios: Union[torch.Tensor, np.ndarray], batch_size: int = 1):
        with torch.no_grad():
            output = self.model(**self.feature_extractor(audios, return_tensors="pt"), output_hidden_states=True)
            embedding = self.psuedo_pooler_output(output)

        return self.check_embedding(embedding)

class MultiModalEmbedder(Embedder):
    def __init__(self, task_name: str, model_name: str):    
        super().__init__()

        self.processor = self.load_processor(model_name)    
        self.logger.info(f"Processor loaded")
        self.model = self.load_model(task_name, model_name)
        self.logger.info(f"Model loaded")

    def load_processor(self, model_name: str):
        return self.load(model_name, [AutoProcessor])
    
    def load_model(self, task_name, model_name):
        #TODO: Add all classes
        from transformers import (
            AutoModelForMaskedImageModeling,
            AutoModelForImageToImage,
            AutoModelForZeroShotObjectDetection,
        )
        classes = {
            "masked-image-modeling": [AutoModelForMaskedImageModeling],
            "image-to-image": [AutoModelForImageToImage],
            "zero-shot-object-detection": [AutoModelForZeroShotObjectDetection],
        }

        return self.load(model_name, classes.get(task_name, [AutoModel]))

    def __call__(self, texts: List[str], images: Union[torch.Tensor, np.ndarray], batch_size: int = 1):   
        with torch.no_grad(): 
            outputs = self.model(**self.processor(text=texts, images=images, return_tensors="pt"))   
            embedding = torch.cat((outputs.text_model_output.pooler_output, outputs.vision_model_output.pooler_output), dim=-1)

        return self.check_embedding(embedding)


