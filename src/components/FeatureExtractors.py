from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor, AutoProcessor, AutoModelForImageClassification
from transformers import AutoModelForDepthEstimation, AutoModelForImageClassification, AutoModelForVideoClassification, AutoModelForMaskedImageModeling, AutoModelForObjectDetection, AutoModelForImageSegmentation, AutoModelForImageToImage, AutoModelForSemanticSegmentation, AutoModelForInstanceSegmentation, AutoModelForUniversalSegmentation, AutoModelForZeroShotImageClassification, AutoModelForZeroShotObjectDetection
from transformers import (
    AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, 
    AutoModelForSequenceClassification, AutoModelForMultipleChoice, 
    AutoModelForNextSentencePrediction, AutoModelForTokenClassification, 
    AutoModelForQuestionAnswering, AutoModelForTextEncoding
)
import torch
import warnings
from transformers import AutoModelForAudioClassification


class TextFeatureExtractor:
    def __init__(self, task_name:str, model_name: str):
        self.feature_extractor = self.load_feature_extractor(model_name)
        self.model = self.load_model(task_name, model_name)

    def load_feature_extractor(self, model_name):
        warnings.simplefilter("error")
        classes = [AutoTokenizer, AutoFeatureExtractor] 
        for class_ in classes:
            try:
                return class_.from_pretrained(model_name)
            except Exception as e:
                pass
            except OSError as e: 
                pass 
        return None


    def load_model(self, task_name, model_name):
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
            ]
            }

        warnings.simplefilter("error")

        for task in classes: 
            if task in task_name: 
                for model_class in classes[task]: 
                    try: 
                        return model_class.from_pretrained(model_name)
                    except: 
                        pass
        return None
        
    def __call__(self, text):
        # Text is a tuple 
        if type(text) != tuple: 
            raise ValueError("Input is not a tuple")
        b = self.feature_extractor(text[0], return_tensors="pt", padding='max_length', truncation=True)
        with torch.no_grad(): 
            c = self.model(**b)
        return c.logits.numpy(), text[1]

class ImageFeatureExtractor:
    def __init__(self, task_name:str, model_name: str):
        self.feature_extractor = self.load_feature_extractor(model_name)
        self.model = self.load_model(task_name, model_name)
    
    def load_feature_extractor(self, model_name): 
        warnings.simplefilter("error")
        try: 
            return AutoFeatureExtractor.from_pretrained(model_name)
        except: 
            try: 
                return AutoImageProcessor.from_pretrained(model_name)
            except: 
                return None
            
    def load_model(self, task_name, model_name): 
        classes = {'depth-estimation': [AutoModelForDepthEstimation], 
            'image-classification': [AutoModelForImageClassification], 
            'video-classification': [AutoModelForVideoClassification], 
            'object-detection': [AutoModelForObjectDetection], 
            'image-segmentation': [AutoModelForImageSegmentation, AutoModelForUniversalSegmentation, AutoModelForSemanticSegmentation, AutoModelForInstanceSegmentation], 
            'zero-shot-image-classification': [AutoModelForZeroShotImageClassification]}
        warnings.simplefilter("error")

        for task in classes: 
            if task in task_name: 
                for model_class in classes[task]: 
                    try: 
                        return model_class.from_pretrained(model_name)
                    except: 
                        pass
        return None
                

    def __call__(self, image):
        # Image is a tuple here
        if type(image) != tuple: 
            print(image)
            raise ValueError("Input is not a tuple")
        b = self.feature_extractor(images=image[0], return_tensors="pt")
        with torch.no_grad(): 
            c = self.model(**b)
        return c.logits.numpy(), image[1]


class AudioFeatureExtractor: 
  def __init__(self, task_name:str, model_name: str):
    self.feature_extractor = self.load_feature_extractor(model_name)
    self.model = self.load_model(task_name, model_name)
  
  def load_feature_extractor(self, model_name): 
    return AutoFeatureExtractor.from_pretrained(model_name)
  
  def load_model(self, task_name, model_name):
    warnings.simplefilter("error")
    classes = {'audio-classification': [AutoModelForAudioClassification, AutoModel]}
    for task in classes: 
        if task in task_name: 
            for model_class in classes[task]: 
                try: 
                    return model_class.from_pretrained(model_name)
                except: 
                    pass
    return None

  def __call__(self, audio):
    # Image is a tuple here
    if type(audio) != tuple: 
        print(audio)
        raise ValueError("Input is not a tuple") 
    try: 
      b = self.feature_extractor(audio[0][1], sampling_rate=audio[0][0], return_tensors="pt")
    except ValueError as e: 
      # Sampling Rate issue
      return None, None

    with torch.no_grad(): 
        c = self.model(**b)
    return c.logits.numpy(), audio[1]